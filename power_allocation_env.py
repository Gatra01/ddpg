import numpy as np

class PowerAllocationEnv:
    def __init__(self, num_nodes=3, max_power=1.0, noise_power=0.1):
        self.num_nodes = num_nodes
        #self.num_devices_per_node = num_devices_per_node
        #self.num_devices = num_nodes * num_devices_per_node  # Total devices dalam sistem
        
        self.max_power = max_power
        self.noise_power = noise_power
        
        # State: [power allocation (t-1), channel gain, data rate (t-1), energy efficiency]
        self.state_dim = 4 * self.num_nodes
        self.action_dim = self.num_nodes  # Action = alokasi daya ke setiap node
        
        self.reset()

    def reset(self):
        """Menginisialisasi ulang environment"""
        self.power_alloc = np.random.uniform(0, self.max_power, self.num_nodes)  # Power awal per node
        self.channel_gain = np.random.rayleigh(scale=1.0, size=self.num_nodes)  # Channel gain per device
        self.data_rate = np.zeros(self.num_nodes)  # Data rate awal
        self.energy_efficiency = np.zeros(self.num_nodes)  # Efisiensi energi awal
        
        self.state = np.concatenate([self.power_alloc.repeat(self.num_nodes), self.channel_gain, self.data_rate, self.energy_efficiency])
        return self.state

    def step(self, action):
        """Menjalankan aksi dan menghitung reward"""
        action = np.clip(action, 0, self.max_power)  # Batasi daya, np clip tu ngga mengurangi anggota array.
        power_per_device = np.repeat(action / self.num_nodes, self.num_nodes)
        
        # Hitung SINR per device
        sinr = (power_per_device * self.channel_gain) / (self.noise_power + 1e-6)
        
        # Hitung data rate berdasarkan Shannon Capacity
        data_rate = np.log2(1 + sinr)
        
        # Hitung efisiensi energi: total data rate / total power yang digunakan
        total_data_rate = np.sum(data_rate)
        total_power_used = np.sum(power_per_device)
        energy_efficiency = total_data_rate / (total_power_used + 1e-6)
        
        # Reward berdasarkan efisiensi energi
        reward = energy_efficiency
        
        # Update state
        self.state = np.concatenate([action.repeat(self.num_nodes),self.channel_gain, data_rate, np.full(self.num_nodes, energy_efficiency)])

        
        done = False  # Bisa diubah berdasarkan kondisi spesifik
        
        return self.state, reward, done, {}

    def render(self):
        """Menampilkan informasi sistem"""
        print(f"Power Allocation: {self.power_alloc}")
        print(f"Channel Gain: {self.channel_gain}")
        print(f"Data Rate: {self.data_rate}")
        print(f"Energy Efficiency: {self.energy_efficiency}")
