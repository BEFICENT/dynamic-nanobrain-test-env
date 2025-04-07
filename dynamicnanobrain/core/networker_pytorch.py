import torch
from torch import nn
from activations import TransistorIVActivation, eta_ABC

class NanoOptoElectronicNeuron(nn.Module):
    ## Assume that N = 1 such that this layer is only for one neuron
    ## Users of this module will have to connect all neurons to each other according
    ## to their own setups. This allows for resevoir implementations
    super().__init__()
    keys = ['Rinh','Rexc','RLED','Rstore','Cinh','Cexc','CLED','Cstore','Cgate','Vt','m','I_Vt','vt','Lg','AB','CB']

    kT = 0.02585
    def __init__(self, N, unity_coefficient = 1.0, **device_params):
        self.N = N
        self.device_params = device_params
        self.A = self.calc_A()
        self.B = self.calc_B()
        self.linslope = self.calc_linslope()
        self.gled = torch.zeros((1,))
        self.Vt_vec = None
        self.Vthres = self.device_params['Vthres']
        if unity_coefficient is None:
            self.unity_coefficient = self.calc_unity_coefficient()
        else:
            self.unity_coefficient = unity_coefficient

    def calc_linslope(self):
        return self.device_params['Cgate']*self.device_params['vt']*1e9 # nA/V    
    
    def calc_A(self, Rstore=None, Cstore=None):
        # Sum the memory and gate capacitance, convert Lg in um to cm
        if Cstore is None:
            Cmem = self.device_params['Cstore'] + self.device_params['Cgate']*self.device_params['Lg']*1e-4 
        else:
            Cmem = Cstore + self.device_params['Cgate']*self.device_params['Lg']*1e-4  
        # System frequencies
        g11 = 1e-9/self.device_params['Cinh']/self.device_params['Rinh'] # ns^-1 # GHz
        g22 = 1e-9/self.device_params['Cexc']/self.device_params['Rexc'] # ns^-1 # GHz
        g13 = 1e-9/Cmem/self.device_params['Rinh'] # ns^-1 # GHz
        g23 = 1e-9/Cmem/self.device_params['Rexc'] # ns^-1 # GHz
        if Rstore is not None :
            g33 = 1e-9/Cmem/Rstore
        else :
            g33 = 1e-9/Cmem/self.device_params['Rstore'] # ns^-1 # GHz
        gled = 1e-9/self.device_params['CLED']/self.device_params['RLED'] # ns^-1 # GHz

        new_gammas = torch.tensor([g11,g22,g13,g23,g33,gled])

        self.gled = new_gammas[-1]

        gsum = g13+g23+g33
        A = torch.tensor([[-g11, 0, g11],
                      [0, -g22, g22],
                      [g13, g23, -gsum]])
        
        return A

    def calc_B(self):
        return torch.diag(torch.tensor([
            1e-18/self.device_params['Cinh'],
            1e-18/self.device_params['Cexc'],
            0.
        ]))

    # Maybe rename this to coupling coefficient?
    def calc_unity_coefficient(self):
        # Remember nA
        Rsum=self.device_params['Rstore']+self.device_params['Rexc']
        max_Vg = self.Vthres*self.device_params['Rstore']/Rsum
        Iexc = self.Vthres/Rsum*1e9 # nA
        Isd = TransistorIVActivation().forward(Vg=max_Vg, 
                                               Vt=self.device_params['Vt'] ,
                                               Vt_vec=None, 
                                               I_Vt=self.device_params["I_Vt"],
                                               kT=NanoOptoElectronicNeuron.kT,
                                               linslope=self.linslope,
                                               m=self.device_params["m"]) 
        Iout = eta_ABC().forward(Isd, AB=self.device_params["AB"], CB=self.device_params["CB"]) * Isd
        return Iexc/Iout, Iexc

    def init_neuron(self):
        return torch.zeros([3, self.N])
    
                
    def forward(self, I_inh, I_exc, V, dt):
        I_t = torch.tensor([I_inh, I_exc, 0])
        # Calculate dV_t = AV + Bscale * B
        dV = torch.matmul(self.A, V) + torch.matmul(self.B, I_t)
        # V = V + dV * dt
        V = V + dV * dt
        # Count voltage overshoots
        overshoots = (V < -self.Vthres) * (V > self.Vthres)
        N = torch.sum(overshoots)

        # Voltage clipping (third try)
        V = torch.clip(self.V,-self.Vthres,self.Vthres)

        self.overshoots += N

        # Calculate source drain current
        ISD = TransistorIVActivation().forward(Vg=V[2], 
                                               Vt=self.device_params['Vt'],
                                               Vt_vec=self.Vt_vec, 
                                               I_Vt=self.device_params["I_Vt"],
                                               kT=NanoOptoElectronicNeuron.kT,
                                               linslope=self.linslope,
                                               m=self.device_params["m"]) 

        # Calclate current through LED
        I_led += dt * self.gled * (ISD - I_t)
        
        # Convert LED current to power through efficiency function
        # efficiency function will also be an nn.Module
        P = I_led * eta_ABC().forward(I_led)

        P_scaled = P * self.unity_coefficient

        return P_scaled

        