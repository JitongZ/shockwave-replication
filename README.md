# Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning

## Team

- Rachel
- Jitong

## Source

Based on the paper "Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning"
Developed on top of [Gavel](https://github.com/stanford-futuredata/gavel)

## TL;DR

We implemented the Shockwave scheduler and its policy to achieve a smaller makespan, comparable job completion time, and a smaller unfair job fraction compared to the paper’s baseline when scheduling ML workloads in simulation. However, we were unable to replicate or reproduce the exact results from the paper.

## Github

Project Repository: [Shockwave Replication](#)

## Key Insights from Paper

Shockwave is a scheduler that optimizes both long-term efficiency and long-term fairness for ML training jobs with elastic resource requirements on a shared cluster. It extends market theory to discrete-time and dynamic settings to support dynamic adaptation, providing forecasts for the trajectory of dynamic adaptation in the market.

### Problem with Existing Scheduling Mechanisms

- Existing fair-share algorithms that guarantee fair share restrictively at each instant can degrade long-term efficiency.
- Some schemes filter jobs with the least resource share in the past, maximizing efficiency among these jobs, but a fixed filter is not optimal for completion time.
- Existing systems do not handle dynamic changes in the batch size of training jobs, leading to worse fairness and efficiency.

## Replication Goal

Replicate the result that Shockwave schedules jobs efficiently and fairly on large clusters of different sizes, focusing on Figure 9 from the paper, which shows improvements in makespan and fairness, and retains comparable job completion time against other schedulers when tested at a large scale.

### Evaluation Results

- Shockwave achieves 1.3-1.34x speedup in makespan.
- 2.4x better worst-case Finish Time Fairness.
- Similar Job Completion Time compared with Gavel.

## Methodology of Paper

### Market Theory Application

Shockwave applies classic market theory to dynamic settings to co-optimize efficiency and fairness using the Volatile Fisher Market (VFM), which can reach market equilibrium by maximizing Nash social welfare over time.

### Dynamic Adaptation Prediction

Uses Bayesian statistics to predict dynamic changes in resource demands, incorporating these predictions into the scheduling process.

### System Design and Implementation

1. **Scheduler Solver**: Solves a generalized Nash social welfare optimization problem that includes long-term fairness and efficiency objectives.
2. **Dynamic Adaptation Predictor**: Uses Bayesian models to predict future dynamic changes in jobs, integrating these predictions into scheduling decisions.
3. **Long-Term Efficiency and Fairness Estimators**: Provide inputs for the solver based on predicted job completion times and resource allocation patterns.

### Simulation and Real-World Validation

Validated through trace-driven simulation and real-world cluster experiments comparing Shockwave against existing schedulers on metrics like makespan, fairness, and average job completion time.

## Methodology of Replication

We started with Gavel’s codebase and focused on implementing three main components:

1. Shockwave’s Metadata Collection Module and Integration with Gavel’s Scheduler
2. Expanding Gavel’s Scheduler to Incorporate Prediction of Remaining Runtime Based on Dynamic Batch Size Scaling
3. Implementation of Shockwave Schedule Solver Using CVXPY with GUROBI

### Experimental Setup

- Python for implementation.
- Experiments conducted on Linux machines using CPU resources.
- Simulation experiments due to resource limitations and feasibility of replication.

## Result and Comparison

Comparing our implementation, reproduced scripts, and extracted data from Figure 9 of the paper, we found similar trends in makespan and average job completion time but differences in Fairness Time Factor (FTF) and unfair job fraction. Discrepancies could be due to metric definitions, hyperparameter tuning, trace selection, and implementation details.

## Discussion

Potential factors for discrepancies include:

- **Metric Definition**: Lack of detailed explanations and sample scripts for generating figures.
- **Hyperparameter Tuning**: Uncertainty about exact hyperparameters used in the paper.
- **Trace Selection**: Variations in statistics across different traces.
- **Implementation Details**: Differences in implementation details and approximations needed for DCP compliance.

## Takeaways

This project enhanced our understanding of convex optimization, log approximation techniques, fairness, and efficiency. We learned the importance of detailed reporting and step-by-step guidance in research papers for reproducibility.

## Acknowledgment

We thank the authors of Shockwave for their clarifications and advice, and our teaching teams for their support throughout the project.

## References

1. Zheng, P., et al. Shockwave: Fair and Efficient Cluster Scheduling for Dynamic Adaptation in Machine Learning. USENIX NSDI, 2023.
2. Narayanan, D., et al. Heterogeneity-aware Cluster Scheduling Policies for Deep Learning Workloads. USENIX OSDI, 2020.
3. Branzei, S., et al. Nash Social Welfare Approximation for Strategic Agents. ACM EC, 2017.
4. Mahajan, K., et al. Themis: Fair and Efficient GPU Cluster Scheduling. USENIX NSDI, 2020.
5. Disciplined Convex Programming. [CVXPY Documentation](https://www.cvxpy.org/tutorial/dcp/).
6. Agarwal, S., et al. Adaptive Gradient Communication via Critical Learning Regime Identification. Machine Learning and Systems 3, 2021.
7. Qin, H., et al. Simigrad: Fine-grained Adaptive Batching for Large Scale Training Using Gradient Similarity Measurement. Advances in Neural Information Processing Systems 34, 2021.
