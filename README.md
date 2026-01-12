# Improving Energy Efficiency in Federated Learning Through Resource Scheduling Optimization of Wireless IoT Networks
Federated Learning (FL) has become a key enabler of distributed intelligence at the network edge, allowing multiple devices to collaboratively train a global model while preserving data privacy. However, in wireless IoT networks, the performance of FL is severely limited by unreliable communication channels, energy constraints, and heterogeneous data distributions across devices. These factors often lead to unstable convergence, reduced accuracy, and excessive energy consumption. To address these challenges, this work introduces FL-E2WS, an energy-efficient FL framework that jointly optimizes device participation and allocation of the uplink communication resources. The proposed approach integrates statistical awareness and communication efficiency, selecting devices that contribute most effectively to model convergence while minimizing transmission costs. The proposed solution enhances the performance of FL in wireless networks by improving model accuracy, reducing energy expenditure, and ensuring a fair utilization of communication and computation resources.

---

## 🎯 Contributions
 
- **FL-aware network integration:**  
FL-E2WS incorporates wireless IoT network models for power, bandwidth, and CPU frequency allocation into the FL process, enabling energy-aware and communication-efficient training.

- **Three-stage optimization design:**  
The algorithm unifies device selection based on EMD and SINR with a MILP-based resource scheduling policy for uplink optimization and an aggregation stage weighted by both the amount of local data and its quality.

- **Problem formalization:**  
The MILP formulation proposed can be directly solved using standard optimization solvers, such as the PuLP library, yielding an exact solution within the scope of the proposed model.

- **Performance and insights:**  
FL-E2WS is compared with three baseline algorithms that adopt different power control policies and distinct RBs allocation strategies. Experimental results demonstrate that FL-E2WS algorithm surpasses the baseline methods by achieving higher accuracy while significantly reducing the energy cost, owing to its ability to select the most suitable devices for training and to optimize communication and computation resources efficiently.

- **Implementation and reproducibility:** 
The source code, dataset generation scripts, and simulation parameters are publicly available, ensuring full reproducibility of the results and enabling further extensions of the proposed framework.
