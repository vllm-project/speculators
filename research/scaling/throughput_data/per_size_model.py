import numpy

def expected_acceptance_length(alpha, draft):
    if isinstance(alpha, numpy.ndarray):
        a1 = numpy.where(alpha < 1.0)
        E = draft + 1
        E[a1] = (1. - alpha[a1] ** (draft[a1] + 1)) / (1. - alpha[a1])
    else:
        if alpha == 1.0:
            E = draft + 1
        else:
            E = (1. - alpha ** (draft + 1)) / (1. - alpha)
    return E

class PerSizeModel:
    def __init__(self):
        self.ellpw = None
        self.ellvw = None
        self.elldw = None
        self.ellpB = None
        self.ellvB = None
        self.elldB = None
 
    def batch_size(self, k, alpha, rps):
        return rps * self.latency(k, alpha, rps)

    def ttft(self, k, alpha, rps):
        return self.ellpB * self.batch_size(k, alpha, rps) + self.ellpw
    
    def itl(self, k, alpha, rps):
        B = self.batch_size(k, alpha, rps)
        E = expected_acceptance_length(alpha, k)
        return (B*(self.ellvB + k * self.elldB) +  self.ellvw +k * self.elldw)/E

    def latency(self, k, alpha, rps):
        E = expected_acceptance_length(alpha, k)
        return (self.ellpw + (self.ellvw + k * self.elldw)/E) / (1. - rps * (self.ellpB + (self.ellvB + k * self.elldB)/E))

    def fit_model(self, k, alpha, rps, reference_latency):
        E = expected_acceptance_length(alpha, k)
        A = numpy.vstack(
            [

                1./reference_latency,
                1./(E*reference_latency),
                k/(E*reference_latency),
                rps,
                rps/E,
                rps*k/E
            ]
        ).T
        z = numpy.ones_like(rps)
        coeffs = numpy.linalg.lstsq(A, z, rcond=None)[0]   # z ≈ a + b x
        self.ellpw = coeffs[0]
        self.ellvw = coeffs[1]
        self.elldw = coeffs[2]
        self.ellpB = coeffs[3]
        self.ellvB = coeffs[4]
        self.elldB = coeffs[5]
