#include <vector>
#include <random>
#include <cmath>

static std::mt19937 rng(42);
static std::uniform_real_distribution<double> uni(0.0, 1.0);

extern "C" {

// Hexagonal neighbors (periodic)
int neighbor(int i, int dx, int dy, int L) {
    int x = i / L;
    int y = i % L;
    int nx = (x + dx + L) % L;
    int ny = (y + dy + L) % L;
    return nx * L + ny;
}

double energy(const std::vector<int>& cfg, int L, double J) {
    static int shifts[6][2] = {{-1,0},{1,0},{0,-1},{0,1},{-1,1},{1,-1}};
    double E = 0.0;
    for (int i = 0; i < L*L; ++i) {
        if (!cfg[i]) continue;
        for (auto& s : shifts) {
            int j = neighbor(i, s[0], s[1], L);
            if (cfg[j]) E += J;
        }
    }
    return 0.5 * E;
}

void metropolis_step(
    int* config, int L, double J, double T, int steps
) {
    const double kB = 8.617333e-5;
    double beta = 1.0 / (kB * T);

    std::vector<int> cfg(config, config + L*L);

    for (int s = 0; s < steps; ++s) {
        int i = rng() % (L*L);
        cfg[i] ^= 1;

        double dE = energy(cfg, L, J) -
                    energy(std::vector<int>(config, config + L*L), L, J);

        if (dE < 0 || uni(rng) < std::exp(-beta * dE)) {
            config[i] = cfg[i];
        } else {
            cfg[i] ^= 1; // revert
        }
    }
}
}
