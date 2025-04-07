import torch

class TransistorIVActivation(torch.nn.Module):
    def forward(self, Vg, Vt, Vt_vec, I_Vt, kT, linslope, m):
        if Vt_vec is None :
            mask = Vg < Vt
            return torch.where(mask, Id_sub_0()(Vg=Vg, 
                                                Vt=Vt, 
                                                I_Vt=I_Vt, 
                                                m=m, 
                                                kT=kT), 
                                    Id_sat_0()(Vg=Vg, 
                                               Vt=Vt, 
                                               I_Vt=I_Vt, 
                                               linslope=linslope))
        else:
            Vt = Vt_vec
            mask = Vg < Vt
            return torch.where(mask, 
                               Id_sub().forward(Vg=Vg, 
                                                    Vt=Vt, 
                                                    mask=mask,
                                                    I_Vt=I_Vt,
                                                    m=m,
                                                    kT=kT), 
                               Id_sat()(Vg=Vg,
                                        Vt=Vt,
                                        mask=mask,
                                        I_Vt=I_Vt,
                                        linslope=linslope))   
    
class Id_sub(torch.nn.Module):
    def forward(self, Vg, Vt, mask, I_Vt, m, kT) :
        Vt_masked = Vt * mask
        return I_Vt*torch.exp((Vg-Vt_masked)/m/kT)

class Id_sat(torch.nn.Module):
    def forward(self,Vg,Vt,mask, I_Vt, linslope) :
        # Invert the mask for this one
        Vt_masked = Vt * (1 - mask)
        return I_Vt + linslope*(Vg-Vt_masked)

class Id_sub_0(torch.nn.Module):
    def forward(self,Vg,Vt, I_Vt, m, kT) :
        return I_Vt*torch.exp((Vg-Vt)/m/kT)

class Id_sat_0(torch.nn.Module):
    def forward(self,Vg,Vt, I_Vt, linslope) :
        return I_Vt + linslope * (Vg-Vt)

class eta_ABC(torch.nn.Module):
    def forward(self, I, AB, CB):
        I_scaled = I * 1e-3
        return I_scaled/(AB + I_scaled + CB*I_scaled**2)

