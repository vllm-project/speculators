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

class PerUseCaseModel:
    def __init__(self):
        self.ellpwc = None
        self.ellpwp = None
        self.ellvwc = None
        self.ellvwp = None
        self.elldwc = None
        self.elldwd = None
        self.ellpBc = None
        self.ellpBp = None
        self.ellpBh = None
        self.ellvBc = None
        self.ellvBp = None
        self.ellvBh = None
        self.elldBc = None
        self.elldBd = None
        self.elldBh = None
 
    def ellpw(self, verifier_size):
        return self.ellpwc + self.ellpwp * verifier_size

    def ellvw(self, verifier_size):
        return self.ellvwc + self.ellvwp * verifier_size

    def elldw(self, drafter_size):
        return self.elldwc + self.elldwd * drafter_size
    
    def ellpB(self, verifier_size, hidden_size, number_of_layers):
        return self.ellpBc + self.ellpBp * verifier_size + self.ellpBh * hidden_size * number_of_layers
    
    def ellvB(self, verifier_size, hidden_size, number_of_layers):
        return self.ellvBc + self.ellvBp * verifier_size + self.ellvBh * hidden_size * number_of_layers
    
    def elldB(self, drafter_size, hidden_size):
        return self.elldBc + self.elldBd * drafter_size + self.elldBh * hidden_size

    def batch_size(self, k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers):
        return rps * self.latency(k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers)

    def ttft(self, k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers):
        ellpB = self.ellpB(verifier_size, hidden_size, number_of_layers)
        ellpw = self.ellpw(verifier_size)
        B = self.batch_size(k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers)
        return ellpB * B + ellpw
    
    def itl(self, k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers):
        ellvB = self.ellvB(verifier_size, hidden_size, number_of_layers)
        elldB = self.elldB(drafter_size, hidden_size)
        ellvw = self.ellvw(verifier_size)
        elldw = self.elldw(drafter_size)
        B = self.batch_size(k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers)
        E = expected_acceptance_length(alpha, k)
        return (B*(ellvB + k * elldB) +  ellvw +k * elldw)/E

    def latency(self, k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers):
        ellpw_ = self.ellpw(verifier_size)
        ellvw_ = self.ellvw(verifier_size)
        elldw_ = self.elldw(drafter_size)
        ellpB_ = self.ellpB(verifier_size, hidden_size, number_of_layers)
        ellvB_ = self.ellvB(verifier_size, hidden_size, number_of_layers)
        elldB_ = self.elldB(drafter_size, hidden_size)
        E = expected_acceptance_length(alpha, k)
        return (ellpw_ + (ellvw_ + k * elldw_)/E) / (1. - rps * (ellpB_ + (ellvB_ + k * elldB_)/E))

    def fit_model(self, k, alpha, rps, verifier_size, drafter_size, hidden_size, number_of_layers, reference_latency):
        E = expected_acceptance_length(alpha, k)
        A = numpy.vstack(
            [
                1./reference_latency,
                verifier_size/reference_latency,              
                1./(E*reference_latency),
                verifier_size/(E*reference_latency),
                k/(E*reference_latency),
                drafter_size*k/(E*reference_latency),
                rps,
                rps*verifier_size,
                rps*number_of_layers*hidden_size,                
                rps/E,
                rps*verifier_size/E,
                rps*number_of_layers*hidden_size/E,
                rps*k/E,
                rps*drafter_size*k/E,
                rps*hidden_size*k/E,
            ]
        ).T
        z = numpy.ones_like(rps)
        coeffs = numpy.linalg.lstsq(A, z, rcond=None)[0]   # z ≈ a + b x
        self.ellpwc = coeffs[0]
        self.ellpwp = coeffs[1]
        self.ellvwc = coeffs[2]
        self.ellvwp = coeffs[3]
        self.elldwc = coeffs[4]
        self.elldwd = coeffs[5]
        self.ellpBc = coeffs[6]
        self.ellpBp = coeffs[7]
        self.ellpBh = coeffs[8]
        self.ellvBc = coeffs[9]
        self.ellvBp = coeffs[10]
        self.ellvBh = coeffs[11]
        self.elldBc = coeffs[12]
        self.elldBd = coeffs[13]
        self.elldBh = coeffs[14]
