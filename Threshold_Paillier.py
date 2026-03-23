import random
import math
import gmpy2
import os
import json
import time
from pathlib import Path
from typing import List, Optional

class ThresholdPaillierCorrectProtocol:
    
    def __init__(self, threshold: int = 2, num_parties: int = 3, verbose: Optional[bool] = None):
        self.t = threshold  
        self.l = num_parties 
        self.key_length = int(os.getenv("TP_KEY_BITS", "2048"))
        env_verbose = os.getenv("TP_VERBOSE", "0")
        self.verbose = self._coerce_bool(env_verbose) if verbose is None else bool(verbose)
        self._progress_every = max(1, int(os.getenv("TP_PROGRESS_EVERY", "100")))
        self._started_at = time.perf_counter()
        self._system_random = random.SystemRandom()
        self._gmp_rand_state = gmpy2.random_state(self._system_random.getrandbits(256))
        self._mu_cache = {}
        self._key_cache_path = Path(__file__).with_name(f"tp_key_cache_{self.key_length}.json")
        self._log(f"Protocol init: threshold={self.t}, parties={self.l}, key_bits={self.key_length}")
        self.key_setup()

    def _coerce_bool(self, value) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _log(self, message: str):
        if not self.verbose:
            return
        elapsed = time.perf_counter() - self._started_at
        print(f"[TP +{elapsed:8.2f}s] {message}")

    def key_setup(self):
        key_length = self.key_length
        p_bar_bits = key_length - 1
        self._log("Starting key setup")

        if self._load_cached_key_material():
            self._log("Using cached key material")
            return

        self._log("No valid cache found, generating fresh safe primes")

        def generate_safe_prime(target_bits, label):
            attempts = 0
            started = time.perf_counter()
            while True:
                attempts += 1
                if attempts % self._progress_every == 0:
                    elapsed = time.perf_counter() - started
                    self._log(f"Generating {label}: attempts={attempts}, elapsed={elapsed:.2f}s")
                candidate_bar = gmpy2.mpz_urandomb(self._gmp_rand_state, p_bar_bits)
                candidate_bar |= (gmpy2.mpz(1) << (p_bar_bits - 1))
                candidate_bar |= gmpy2.mpz(1)
                if not gmpy2.is_prime(candidate_bar):
                    continue
                candidate = gmpy2.mpz(2) * candidate_bar + 1
                if gmpy2.is_prime(candidate) and candidate.bit_length() == target_bits:
                    elapsed = time.perf_counter() - started
                    self._log(f"Generated {label} in {attempts} attempts ({elapsed:.2f}s)")
                    return candidate_bar, candidate

        self.p_bar, self.p = generate_safe_prime(key_length, "p")
        self.q_bar, self.q = generate_safe_prime(key_length, "q")
        while self.q == self.p:
            self._log("q matched p; regenerating q")
            self.q_bar, self.q = generate_safe_prime(key_length, "q")

        while (self.p * self.q).bit_length() < 2 * key_length:
            self._log("n bit-length too small; regenerating q")
            self.q_bar, self.q = generate_safe_prime(key_length, "q")
            while self.q == self.p:
                self._log("q matched p during n-size check; regenerating q")
                self.q_bar, self.q = generate_safe_prime(key_length, "q")

        self._log("Prime generation complete, finalizing key material")
        self._finalize_key_material()
        self._log("Key setup complete")

    def _finalize_key_material(self):
        self._log("Computing derived key material")
        self.n = self.p * self.q
        self.m = self.p_bar * self.q_bar
        self.n_squared = self.n * self.n
        self.nm = self.n * self.m

        phi_n = (self.p - 1) * (self.q - 1)
        if gmpy2.gcd(self.n, phi_n) != 1:
            raise ValueError("gcd(n, phi_n) != 1; choose different primes.")
        
        self.beta = self.select_from_group(self.n, star=True)
        self.a = self.select_from_group(self.n, star=True)
        self.b = self.select_from_group(self.n, star=True)
        
        self.g = gmpy2.f_mod(
            gmpy2.powmod(1 + self.n, self.a, self.n_squared) * gmpy2.powmod(self.b, self.n, self.n_squared),
            self.n_squared
        )

        self.SK = gmpy2.f_mod(self.beta * self.m, self.nm)

        self.a_i = self.shamir_secret_sharing()
        self.s_i = self.compute_secret_shares()

        self.delta = math.factorial(self.l)
        self.two_delta = 2 * self.delta
        self.theta = gmpy2.f_mod(self.a * self.m * self.beta, self.n)
        denominator = gmpy2.f_mod(4 * self.delta * self.delta * self.theta, self.n)
        self._denominator_inv = gmpy2.invert(denominator, self.n)
        self.v = gmpy2.powmod(self.b, 2, self.n_squared)
        # self.VK_i = []
        # for i in range(self.l):
        #     vk_i = gmpy2.powmod(self.v, self.delta * self.s_i[i], self.n_squared)
        #     self.VK_i.append(vk_i)
        self.VK_i = [gmpy2.powmod(self.v, self.delta * s, self.n_squared) for s in self.s_i]
        self._save_cached_key_material()
        self._log("Derived material ready and cache saved")

    def _load_cached_key_material(self):
        if not self._key_cache_path.exists():
            self._log("Key cache file not found")
            return False
        try:
            self._log("Loading key cache file")
            with self._key_cache_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("key_length") != self.key_length:
                self._log("Key cache key length mismatch")
                return False

            self.p_bar = gmpy2.mpz(payload["p_bar"])
            self.q_bar = gmpy2.mpz(payload["q_bar"])
            self.p = gmpy2.mpz(payload["p"])
            self.q = gmpy2.mpz(payload["q"])

            if self.p.bit_length() != self.key_length or self.q.bit_length() != self.key_length:
                self._log("Key cache prime size mismatch")
                return False
            if not gmpy2.is_prime(self.p) or not gmpy2.is_prime(self.q):
                self._log("Key cache prime validation failed")
                return False
            if self.p != 2 * self.p_bar + 1 or self.q != 2 * self.q_bar + 1:
                self._log("Key cache safe-prime relation failed")
                return False

            self._finalize_key_material()
            return True
        except Exception:
            self._log("Failed to load key cache file")
            return False

    def _save_cached_key_material(self):
        payload = {
            "key_length": self.key_length,
            "p_bar": str(self.p_bar),
            "q_bar": str(self.q_bar),
            "p": str(self.p),
            "q": str(self.q),
        }
        try:
            with self._key_cache_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
        except Exception:
            pass

    def shamir_secret_sharing(self) -> List[gmpy2.mpz]:
        a = [gmpy2.mpz(self.SK)]
        for i in range(1, self.t):
            a_i = gmpy2.mpz(gmpy2.mpz_random(self._gmp_rand_state, self.nm))
            a.append(a_i)
        return a
    
    def compute_secret_shares(self) -> List[gmpy2.mpz]:
        secret_shares = []
        for i in range(1, self.l + 1):
            s_i = gmpy2.mpz(0)
            for coeff in reversed(self.a_i):
                s_i = (s_i * i + coeff) % self.nm
            secret_shares.append(s_i)
        return secret_shares

    def encryption(self, M: int) -> int:
        self._log(f"Encrypting message: M={M}")
        x = self.select_from_group(self.n, star=True)
        g_pow_m = gmpy2.powmod(self.g, int(M), self.n_squared)
        ciphertext = gmpy2.f_mod(
            g_pow_m * gmpy2.powmod(x, self.n, self.n_squared),
            self.n_squared
        )
        self._log("Encryption complete")
        return int(ciphertext)

    def share_decryption(self, c: int, i: int) -> gmpy2.mpz:
        if i < 1 or i > self.l:
            raise ValueError(f"Invalid party index {i}")
        s_i = self.s_i[i-1]
        c = gmpy2.mpz(c)
        self._log(f"Computing share decryption for party {i}")
        c_i = gmpy2.powmod(c, self.two_delta * s_i, self.n_squared)
        return c_i
    
    def combining_algorithm(self, c: int, participating_parties: List[int]) -> int:
        if len(participating_parties) < self.t:
            raise ValueError(f"Need at least {self.t} parties, got {len(participating_parties)}")
        S = tuple(participating_parties[:self.t])
        self._log(f"Combining shares from parties={list(S)}")

        c_shares = [self.share_decryption(c, i) for i in S]

        mu_coefficients = self._mu_cache.get(S)
        if mu_coefficients is None:
            coeffs = []
            for j in S:
                mu_j = gmpy2.mpz(self.delta)
                for j_prime in S:
                    if j_prime != j:
                        mu_j = mu_j * j_prime // (j_prime - j)
                coeffs.append(mu_j)
            mu_coefficients = tuple(coeffs)
            self._mu_cache[S] = mu_coefficients
            
        product = gmpy2.mpz(1)
        for c_share, mu_j in zip(c_shares, mu_coefficients):
            term = gmpy2.powmod(c_share, 2 * mu_j, self.n_squared)
            product = gmpy2.f_mod(product * term, self.n_squared)

        L_result = self.l_function(product)
        message = gmpy2.f_mod(L_result * self._denominator_inv, self.n)
        self._log("Combine/decryption complete")
        return int(message)

    def l_function(self, u: int) -> gmpy2.mpz:
        u = gmpy2.mpz(u)
        return gmpy2.f_div(gmpy2.sub(u, 1), self.n)

    def select_from_group(self, n: int, star: bool) -> gmpy2.mpz:
        n = gmpy2.mpz(n)
        if not star:
            return gmpy2.mpz(gmpy2.mpz_random(self._gmp_rand_state, n))
        else:
            while True:
                x = gmpy2.mpz(gmpy2.mpz_random(self._gmp_rand_state, n - 1)) + 1
                if gmpy2.gcd(x, n) == 1:
                    return x
 
def test_correct_protocol():
    print("Testing Corrected Threshold Paillier Protocol")
    print("="*60)
    # Set verbose=True for live progress updates.
    system = ThresholdPaillierCorrectProtocol(threshold=4, num_parties=4, verbose=True)
    message = 5555555555555
    ciphertext = system.encryption(message)
    print(f"\nEncryption Test:")
    print(f"Original message: {message}")
    print(f"Ciphertext: {ciphertext}")
    participating_parties = [1, 2, 3, 4]
    decrypted = system.combining_algorithm(ciphertext, participating_parties)
    print(f"\nDecryption Test:")
    print(f"Participating parties: {participating_parties}")
    print(f"Decrypted message: {decrypted}")
    print(f"Correct: {message == decrypted}")
    if message == decrypted:
        print("\n SUCCESS: Protocol implemented correctly!")
    else:
        print("\n FAILURE: Implementation has errors")

if __name__ == "__main__":
    test_correct_protocol()
