from itertools import product

N = 3584

kind_to_fc_parameters = {
    "Sensor_1": {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]},
    "Sensor_2": {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]},
    "Sensor_3": {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]},
    "Sensor_4": {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]},
    "Sensor_5": {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]},
    "Sensor_6": {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]},
}

default_fc_parameters = {"fft_coefficient": [{"coeff": k, "attr": a} for a, k in product(["real", "imag", "abs", "angle"], range(N // 2))]}
