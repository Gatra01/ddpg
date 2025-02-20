import numpy as np

class PowerAllocationEnv:
    def __init__(self, num_nodes=2, noise_power=0.01, gamma=0.01, beta=5):
        self.num_nodes = num_nodes
        self.p_max = num_nodes
        self.noise_power = noise_power
        self.gamma= gamma
        self.beta=beta
        
        # Inisialisasi daya, channel gain, SINR, dan data rate
        self.power = np.zeros(self.num_nodes)
        self.channel_gain = np.zeros((self.num_nodes, self.num_nodes))
        self.sinr = np.zeros(self.num_nodes)
        self.data_rate = np.zeros(self.num_nodes)

        # Generate daya & channel gain saat objek dibuat
        self.generate_power()
        self.generate_channel_gain()

    def generate_power(self):
        """Menghasilkan daya yang dialokasikan untuk setiap node"""
        jmlh=0
        for i in range(self.num_nodes):
            self.power[i]=np.random.uniform(0,self.p_max-jmlh)
            jmlh+=self.power[i]
        self.total_daya=np.sum(self.power)
        return self.power
    
    def generate_channel_gain(self):
        """Menghasilkan channel gain menggunakan distribusi Rayleigh"""
        self.channel_gain = np.random.rayleigh(scale=1, size=(self.num_nodes, self.num_nodes))
    
    def hitung_sinr(self):
        """Menghitung SINR untuk semua node"""
        for node_idx in range(self.num_nodes):
            sinr_numerator = (abs(self.channel_gain[node_idx][node_idx]) ** 2) * self.power[node_idx]
            sinr_denominator = self.noise_power + np.sum(
                [(abs(self.channel_gain[node_idx][i]) ** 2) * self.power[i] for i in range(self.num_nodes) if i != node_idx]
            )
            self.sinr[node_idx] = sinr_numerator / sinr_denominator
        return self.sinr 
    
    def hitung_data_rate(self):
        """Menghitung data rate berdasarkan SINR"""
        self.hitung_sinr()
        self.data_rate = np.log(1 + self.sinr)

    def hitung_efisiensi_energi(self):
        """Menghitung efisiensi energi total"""
        total_power = np.sum(self.power)
        total_rate = np.sum(self.data_rate)
        self.EE=total_rate / total_power if total_power > 0 else 0

        return self.EE  # Menghindari pembagian dengan nol

    def observasi(self) :# Pastikan semua nilai telah diperbarui
        obs = np.vstack((self.power, self.data_rate)).tolist()
        return obs
    def observasi2(self):
        return self.generate_power()
    def reward(self) :
        self.reward = self.EE+np.sum(((np.array(self.data_rate)-self.gamma)*self.beta).tolist())+ self.beta*(self.total_daya-self.p_max)
        return self.reward
    def reset(self):
        """Reset environment untuk episode baru"""
        self.generate_power()
        self.generate_channel_gain()
        self.hitung_data_rate()
        return self.observasi()  # Mengembalikan state awal
    def run_simulation(self):
        """Menjalankan seluruh proses perhitungan"""
        self.hitung_data_rate()
        return {
            "power": self.power,
            "channel_gain": self.channel_gain,
            "sinr": self.sinr,
            "data_rate": self.data_rate,
            "energy_efficiency": self.hitung_efisiensi_energi(),
            "obs":self.observasi(),
            "totalpower":self.total_daya,
            "reward" : self.reward(),
            'reset' : self.reset()
            #"step" : self.step()
        }

# Contoh penggunaan

#result = env.run_simulation()

# Menampilkan hasil
'''print("Power:\n", result["power"])
print("Channel Gain:\n", result["channel_gain"])
print("SINR:\n", result["sinr"])
print("Data Rate:\n", result["data_rate"])
print("Energy Efficiency:", result["energy_efficiency"])
#print("obs:", result["obs"])
print("tot power",result["totalpower"])
print("reward",result["reward"])
print("reset",result["reset"])
print("step",result["step"])'''
efisiensi_energi=[0 for i in range(5)]
env = PowerAllocationEnv(5, 5, 0.01)
print(env.hitung_efisiensi_energi())

#print(min(efisiensi_energi))
#print(max(efisiensi_energi))


