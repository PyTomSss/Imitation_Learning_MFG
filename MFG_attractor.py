import numpy as np
from dataclasses import dataclass, field

@dataclass
class AttractorMFG:
    H: int                 # horizon
    L: float               # paramètre Lipschitz
    rho0: float = 0.0      # ρ0(s1)  
    hist_E_pa: list = field(init=False, default_factory=list)
    hist_pa:   list = field(init=False, default_factory=list)

    # ------------- Policy -------------
    @staticmethod
    def policy_prob(alpha: float) -> float:
        return np.clip(alpha, 0.0, 1.0)

    # ------------- Dynamics -------------
    def update_rho(self, rho_s1: float, alpha: float, variant: str) -> float:
        """
        Met à jour rho_n(s_1) selon :
        - variant == 'E_pa'  
        - variant == 'pa'    
        """        
        if variant == "E_pa":
            prob_s1 = alpha                             
        elif variant == "pa":
            prob_s1 = alpha + (1 - alpha) * min(1.0, self.L * rho_s1)
        else:
            raise ValueError("variant must be 'E_pa' or 'pa'")

        return rho_s1 + (1 - rho_s1) * prob_s1          

    # ------------- Simulation -------------
    def run(self, alpha: float):
        """Calcule les trajectoires rho_n pour n = 0…H"""
        self.hist_E_pa = [self.rho0]                     
        self.hist_pa   = [self.rho0]

        for _ in range(self.H):
            self.hist_E_pa.append(
                self.update_rho(self.hist_E_pa[-1], alpha, "E_pa")
            )
            self.hist_pa.append(
                self.update_rho(self.hist_pa[-1],   alpha, "pa")
            )

    # ------------- Compute Errors -------------
    def errors(self, alpha: float):
        """
        Retourne trois tableaux longueur H+1 :
            - ε_BC       
            - ε_vanilla   
            - ε_MFC       
        """
        #if not self.hist_pa:       # s’assurer que run() a été appelé
            #self.run(alpha)

        self.run(alpha)  # recalcul systématique

        bc  = np.full(self.H+1, 2*alpha)
        van = 2*(alpha + (1-alpha)*np.array(self.hist_E_pa))
        mfc = 2*(alpha + (1-alpha)*np.array(self.hist_pa))

        return bc, van, mfc

    def NIG(self, alpha: float):
        """
        Retourne le nombre d’agents dans l’état s_1 à l’horizon H.
        """
        #if not self.hist_pa:       # s’assurer que run() a été appelé
        #   self.run(alpha)
        self.run(alpha)

        return np.cumsum(self.hist_pa)