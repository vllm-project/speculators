"""Race one prompt against two vLLM servers side by side.

Left  = the target model on its own.  Right = the same model + a speculator.
Both decode greedily (temperature 0), the two panels must fill with identical
text, and the right one must finish first.

Started by demo_side_by_side.sh, which serves both engines. Stdlib only.
"""

import argparse
import json
import re
import shutil
import sys
import textwrap
import threading
import time
import urllib.request

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
CYAN, YELLOW, GREEN, RED, GREY = (
    "\033[36m",
    "\033[33m",
    "\033[32m",
    "\033[31m",
    "\033[90m",
)
HOME, CLEAR, CLEAR_LINE = "\033[H", "\033[2J", "\033[K"
HIDE_CURSOR, SHOW_CURSOR = "\033[?25l", "\033[?25h"

LEGEND = (
    "time = request sent to last token  ·  tok/s = speed after the first token  ·  "
    "ttft = time to first token  ·  steps = forward passes of the target model  ·  "
    "tok/step = tokens per forward pass, 1.00 without speculation"
)

# Two rounds before the demo starts, so the first prompt someone types is not the
# one paying for lazy initialisation. Different lengths warm different prefills.
WARMUP = [
    ("Say hello.", 32),
    ("Explain how a B-tree index works, with a short example.", 256),
]


def read_idle_counters(url):
    """Counters, read once the engine has nothing in flight.

    A snapshot taken while an earlier request is still finishing would charge
    that request's forward passes to this race.
    """
    while True:
        counters = read_counters(url)
        if not counters.get("vllm:num_requests_running"):
            return counters
        time.sleep(0.05)


def read_counters(url):
    """Every counter on a server's /metrics, summed over labels.

    Per-draft-position counters keep their position, as `name[2]`, so we can see
    how deep into each draft the speculator actually got.
    """
    text = urllib.request.urlopen(f"{url}/metrics").read().decode()
    counters: dict[str, float] = {}
    for name, labels, value in re.findall(r"^(\S+?)\{(.*?)\}\s+(\S+)$", text, re.M):
        position = re.search(r'position="(\d+)"', labels)
        key = f"{name}[{position.group(1)}]" if position else name
        counters[key] = counters.get(key, 0.0) + float(value)
    return counters


class Side:
    """One engine in the race: what it wrote, and what that cost."""

    def __init__(self, name, url, color):
        self.name, self.url, self.color = name, url, color
        self.text = ""
        self.tokens = 0
        self.started = time.perf_counter()
        self.first_token = None
        self.finished = None
        self.error = None
        self.before = self.after = {}  # /metrics, snapshotted around the race

    @property
    def seconds(self):
        return (self.finished or time.perf_counter()) - self.started

    @property
    def tokens_per_second(self):
        """Decode speed, which starts at the first token: prefill is ttft."""
        if not self.first_token or self.tokens < 2:
            return 0.0
        end = self.finished or time.perf_counter()
        return (self.tokens - 1) / (end - self.first_token)

    @property
    def ttft(self):
        return self.first_token - self.started if self.first_token else None

    def counted(self, name):
        """How far a /metrics counter moved during this race, 0 until it ends."""
        if not self.after:
            return 0.0
        return self.after.get(name, 0.0) - self.before.get(name, 0.0)

    @property
    def steps(self):
        """Forward passes of the target model, from the engine's own counter.

        Counting streamed chunks instead would undercount: vLLM merges several
        steps into one chunk whenever the HTTP layer lags behind the engine.
        """
        # The launcher guarantees one unchunked prefill pass, excluded here.
        total_steps = int(self.counted("vllm:iteration_tokens_total_count"))
        return max(0, total_steps - 1)

    @property
    def tokens_per_step(self):
        return self.tokens / self.steps if self.steps else 0.0

    @property
    def drafting(self):
        """(tokens per draft step, share of drafts used, share by position),
        or None on the engine that is not speculating."""
        drafts = self.counted("vllm:spec_decode_num_drafts_total")
        if not drafts:
            return None
        drafted = self.counted("vllm:spec_decode_num_draft_tokens_total")
        accepted = self.counted("vllm:spec_decode_num_accepted_tokens_total")
        by_position = [
            self.counted(f"vllm:spec_decode_num_accepted_tokens_per_pos_total[{i}]")
            / drafts
            for i in range(int(drafted / drafts))
        ]
        return 1 + accepted / drafts, accepted / drafted, by_position


def stream(side, prompt, max_tokens):
    """Ask one server for a greedy answer and read the tokens as they arrive.

    Both sides are asked for exactly this, from this one place, which is what
    makes the comparison fair. logprobs is on because it carries one entry per
    token, and a speculative step can emit several tokens in a single chunk.
    """
    request = urllib.request.Request(
        f"{side.url}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
                "stream": True,
                "logprobs": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode(),
    )
    try:
        with urllib.request.urlopen(request) as response:
            for line in response:
                if not line.startswith(b"data: ") or line[6:].strip() == b"[DONE]":
                    continue
                choice = json.loads(line[6:])["choices"][0]
                tokens = (choice.get("logprobs") or {}).get("content") or []
                side.text += choice["delta"].get("content") or ""
                side.tokens += len(tokens)
                if tokens and not side.first_token:
                    side.first_token = time.perf_counter()
    except Exception as e:
        side.error = e
    side.finished = time.perf_counter()


# ---- drawing ----------------------------------------------------------------


def visible(text):
    """Width on screen: colour codes take no columns."""
    return len(re.sub(r"\033\[[0-9;?]*[A-Za-z]", "", text))


def fit(text, width):
    return text + " " * (width - visible(text))


def middle(text, width):
    space = width - visible(text)
    return " " * (space // 2) + text + " " * (space - space // 2)


def column(text, width):
    """Text laid out in a column, blank lines kept."""
    lines = []
    for paragraph in text.replace("\t", "    ").split("\n"):
        lines += textwrap.wrap(paragraph, width) or [""]
    return lines


def title(side):
    state = "error" if side.error else ("done" if side.finished else "generating")
    color = RED if side.error else (GREEN if side.finished else side.color)
    return f"{side.color}{BOLD}{side.name}{RESET} {DIM}[{color}{state}{DIM}]{RESET}"


def bar(side, width, max_tokens):
    """One lane of the race: tokens written out of the cap."""
    count = f" {side.tokens:>{len(str(max_tokens))}}/{max_tokens}"
    length = width - len(count)
    filled = round(length * side.tokens / max_tokens)
    color = GREEN if side.finished else side.color
    return f"{color}{'█' * filled}{GREY}{'░' * (length - filled)}{RESET}{count}"


def stats(side):
    def cell(label, value):
        return f"{DIM}{label:<8}{RESET}{BOLD}{value:>7}{RESET}"

    ttft = f"{side.ttft * 1000:.0f}ms" if side.ttft else "-"
    steps = side.steps or "-"  # the counters only land once it is over
    per_step = f"{side.tokens_per_step:.2f}" if side.steps else "-"
    return [
        f"{cell('time', f'{side.seconds:.2f}s')}   "
        f"{cell('tok/s', f'{side.tokens_per_second:.1f}')}   "
        f"{cell('ttft', ttft)}",
        f"{cell('tokens', side.tokens)}   "
        f"{cell('steps', steps)}   "
        f"{cell('tok/step', per_step)}",
    ]


def verdict(left, right, done):
    if not done:
        shared = min(len(left.text), len(right.text))
        if left.text[:shared] != right.text[:shared]:
            return f" {RED}{BOLD}TEXT DIVERGED{RESET}"
        return f" {GREY}text identical so far{RESET}"
    if left.text == right.text:
        result = f"{GREEN}{BOLD}TEXT IDENTICAL{RESET}"
    else:
        result = f"{RED}{BOLD}TEXT DIFFERS{RESET}"
    wall = left.seconds / right.seconds
    speed = right.tokens_per_second / (left.tokens_per_second or 1)
    return (
        f" {result}   {BOLD}{wall:.2f}x{RESET} wall-clock,"
        f" {BOLD}{speed:.2f}x{RESET} tok/s"
    )


def drafting_note(right):
    """What the speculator managed on this prompt, once the race is over."""
    if not right.after:
        return None
    if not right.drafting:
        return (
            f"{YELLOW}no speculation ran"
            f" -- the right-hand engine drafted nothing{RESET}"
        )
    per_step, used, by_position = right.drafting
    return (
        f"{YELLOW}drafter: {per_step:.2f} tokens accepted per step, "
        f"{100 * used:.0f}% of drafted tokens used, accepted by position: "
        + " ".join(f"{100 * share:.0f}%" for share in by_position)
        + RESET
    )


def draw(sides, prompt, max_tokens, done):
    left, right = sides
    columns, rows = shutil.get_terminal_size()
    half = (columns - 3) // 2
    rule = GREY + "─" * columns + RESET

    def both(a, b):
        return f"{fit(a, half)} {GREY}│{RESET} {fit(b, half)}"

    top = [
        f"{BOLD}PROMPT:{RESET} {' '.join(prompt.split())[: columns - 8]}",
        rule,
        both(middle(title(left), half), middle(title(right), half)),
        both(
            " " + bar(left, half - 1, max_tokens),
            " " + bar(right, half - 1, max_tokens),
        ),
        rule,
    ]

    bottom = [rule]
    bottom += [both(" " + a, " " + b) for a, b in zip(stats(left), stats(right))]
    bottom += [rule, verdict(left, right, done)]
    note = drafting_note(right) if done else None
    if note:
        bottom += [" " + line for line in column(note, columns - 2)]
    bottom += [f" {DIM}{line}{RESET}" for line in column(LEGEND, columns - 2)]

    height = rows - len(top) - len(bottom)
    answers = [column(side.text, half - 1) for side in sides]
    scroll = max(0, max(len(a) for a in answers) - height)  # follow the longer side
    middle_rows = [
        both(*[" " + (a[i] if i < len(a) else "") for a in answers])
        for i in range(scroll, scroll + height)
    ]

    # Exactly `rows` lines and no trailing newline, so the screen never scrolls.
    sys.stdout.write(
        HOME + (CLEAR_LINE + "\n").join(top + middle_rows + bottom) + CLEAR_LINE
    )
    sys.stdout.flush()


# ---- running ----------------------------------------------------------------


def run_race(prompt, servers, max_tokens, show=True):
    """Both requests in flight at once, counters pinned either side of them."""
    sides = [
        Side(name, url, color) for (name, url), color in zip(servers, (CYAN, YELLOW))
    ]
    for side in sides:
        side.before = read_idle_counters(side.url)
    threads = [
        threading.Thread(target=stream, args=(side, prompt, max_tokens), daemon=True)
        for side in sides
    ]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads):
        if show:
            draw(sides, prompt, max_tokens, done=False)
        time.sleep(0.06)
    for side in sides:
        side.after = read_counters(side.url)
    if show:
        draw(sides, prompt, max_tokens, done=True)
    return sides


def warm_up(servers):
    """Run the real request path a couple of times before anyone is watching.

    The last round doubles as a check that the two engines agree: they compile
    independently and can land on different numeric variants of the same graph,
    which would break the identical-text claim live rather than here.
    """
    print("warming up both engines on the same request path as the race")
    for prompt, max_tokens in WARMUP[:-1]:
        run_race(prompt, servers, max_tokens, show=False)
    prompt, max_tokens = WARMUP[-1]
    left, right = run_race(prompt, servers, max_tokens, show=False)
    for side in (left, right):
        if side.error:
            sys.exit(f"warm-up failed on {side.name}: {side.error}")
    if left.text == right.text:
        print(f"{GREEN}both engines agree token-for-token{RESET}")
    else:
        print(
            f"{RED}{BOLD}the two engines do NOT agree.{RESET} They compiled "
            f"differently; re-run with VLLM_DISABLE_COMPILE_CACHE=1 set, until "
            f"they do, then drop it so the good build stays cached."
        )


def summarise(races):
    print(f"\n{BOLD}SUMMARY{RESET}")
    for prompt, left, right in races:
        print(
            f"  {GREEN}same{RESET}"
            if left.text == right.text
            else f"  {RED}DIFF{RESET}",
            f"{left.seconds / right.seconds:5.2f}x wall-clock  "
            f"{left.tokens_per_second:6.1f} -> {right.tokens_per_second:6.1f} tok/s  "
            f"{right.tokens_per_step:4.2f} tok/step   {' '.join(prompt.split())[:40]}",
        )


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument("--left-url", required=True)
    parse.add_argument("--right-url", required=True)
    parse.add_argument("--left-name", default="WITHOUT spec-dec")
    parse.add_argument("--right-name", default="WITH spec-dec")
    parse.add_argument("--max-tokens", type=int, default=1000)
    args = parse.parse_args()
    servers = [(args.left_name, args.left_url), (args.right_name, args.right_url)]

    warm_up(servers)
    print(
        f"\n{BOLD}Type a prompt to race it on both engines.{RESET} "
        f"{DIM}Empty line quits.{RESET}"
    )
    races = []
    try:
        while True:
            prompt = input(f"\n{BOLD}prompt>{RESET} ").strip()
            if not prompt:
                break
            sys.stdout.write(CLEAR + HIDE_CURSOR)
            left, right = run_race(prompt, servers, args.max_tokens)
            sys.stdout.write(SHOW_CURSOR + "\n")
            races.append((prompt, left, right))
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        sys.stdout.write(SHOW_CURSOR)
    if races:
        summarise(races)


main()
