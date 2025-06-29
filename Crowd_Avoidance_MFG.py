import numpy as np
from dataclasses import dataclass, field

@dataclass
class CrowdAvoidanceMFG:
    """
    Jouet « Crowd-Avoidance » :
      • deux états  s0 (safe) / s1 (crowded),
      • deux actions stay / go,
      • proba de succès d'un aller vers s1 
      • politique paramétrée  

    Les méthodes fournissent :
      • trajectories
      • errors ε_BC, ε_vanilla-ADV, ε_MPC-ADV,
      • Nash Imitation Gap  
    """
    H: int                  
    L: float                 
    rho0: float = 0.0        

    # historiques (remplis par run)
    hist_E_pa: list = field(init=False, default_factory=list)
    hist_pa:   list = field(init=False, default_factory=list)

    # ---------- Policy ----------
    @staticmethod
    def policy_prob(alpha: float) -> float:
        return np.clip(alpha, 0.0, 1.0)

    # ---------- Dynamics ----------
    def _success_prob(self, rho_s1: float) -> float:
        return max(0.0, 1.0 - self.L * rho_s1**2)

    def update_rho(self, rho_s1: float, alpha: float, variant: str) -> float:
        """
        Retourne 
          variant = 'E_pa' 
          variant = 'pa'   
        """
        if variant == "E_pa":
            a = 1.0
        elif variant == "pa":
            a = self.policy_prob(alpha)
        else:
            raise ValueError("variant must be 'E_pa' or 'pa'")

        p_succ = self._success_prob(rho_s1)
        return rho_s1 + (1 - rho_s1) * a * p_succ

    # ---------- Simulation ----------
    def run(self, alpha: float):
        """Calcule les trajectoires ρ_t^{Eπ} et ρ_t^{π} pour t=0…H."""
        self.hist_E_pa = [self.rho0]
        self.hist_pa   = [self.rho0]

        for _ in range(self.H):
            self.hist_E_pa.append(
                self.update_rho(self.hist_E_pa[-1], alpha=alpha, variant="E_pa")
            )
            self.hist_pa.append(
                self.update_rho(self.hist_pa[-1],   alpha=alpha, variant="pa")
            )

    # ---------- Compute erros ----------
    def errors(self, alpha: float):
        """
        Renvoie trois ndarray (longueur H+1) :
          ε_BC, ε_vanilla-ADV, ε_MPC-ADV
        """
        self.run(alpha)  

        bc  = np.full(self.H + 1, 2 * alpha)
        van = 2 * (alpha + (1 - alpha) * np.array(self.hist_E_pa))
        mpc = 2 * (alpha + (1 - alpha) * np.array(self.hist_pa))
        return bc, van, mpc

    # ---------- Nash Imitation Gap ----------
    def NIG(self, alpha: float):
        self.run(alpha)
        return np.sum(self.hist_pa[:-1]) 
