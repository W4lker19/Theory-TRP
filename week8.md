# Week 8: SMC Application to Private Data Mining

<div align="center">

[⬅️ **Week 7**](week7.md) | [**Main**](README.md) | [**Week 9** ➡️](week9.md)

</div>

---

## 🎯 Learning Goals

By the end of this week, you should understand:
- How to apply SMC protocols to machine learning and data mining tasks
- Privacy-preserving classification, clustering, and regression algorithms
- Federated learning and its relationship to SMC
- Practical challenges in deploying private machine learning systems

---

## 📖 Theoretical Content

### Introduction to Private Data Mining

**The Problem:**
Multiple parties want to collaboratively perform data mining tasks (classification, clustering, association rules) without revealing their private datasets to each other.

**Example Scenarios:**
- **Medical Research:** Hospitals collaborate on disease prediction models without sharing patient records
- **Financial Fraud Detection:** Banks jointly detect fraud patterns without revealing customer data
- **Marketing Analytics:** Companies analyze market trends without exposing customer databases
- **Genomic Research:** Research institutions study genetic patterns while protecting individual privacy

**Privacy Requirements:**
1. **Input Privacy:** Raw data remains confidential
2. **Computation Privacy:** Intermediate results are not disclosed
3. **Output Privacy:** Only agreed-upon results are revealed
4. **Pattern Privacy:** Sensitive patterns within data are protected

### Private Classification

**Problem Setup:**
- Party A has training data (X_A, y_A)
- Party B has training data (X_B, y_B)
- Goal: Train classifier on combined data without data sharing
- Result: Both parties get the trained model

**Naive Bayes Classification:**
SMC-friendly due to its additive nature:

```python
class PrivateNaiveBayes:
    def __init__(self, num_parties):
        self.parties = num_parties
        self.feature_counts = {}
        self.class_counts = {}
        
    def secure_training(self, local_datasets):
        # Step 1: Each party computes local statistics
        local_stats = []
        for party_data in local_datasets:
            stats = self.compute_local_statistics(party_data)
            local_stats.append(stats)
            
        # Step 2: Securely aggregate statistics using SMC
        global_feature_counts = self.secure_sum(
            [stats['feature_counts'] for stats in local_stats]
        )
        global_class_counts = self.secure_sum(
            [stats['class_counts'] for stats in local_stats]
        )
        
        # Step 3: Compute probabilities from aggregated counts
        self.feature_probabilities = self.compute_probabilities(
            global_feature_counts, global_class_counts
        )
        
        return self.feature_probabilities
```

### Private Clustering

**K-Means Clustering:**
Iterative algorithm suitable for SMC adaptation:

```python
class PrivateKMeans:
    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        
    def secure_clustering(self, distributed_datasets):
        # Step 1: Initialize centroids (can be done publicly)
        centroids = self.initialize_centroids(self.k)
        
        for iteration in range(self.max_iterations):
            # Step 2: Assign points to clusters (locally)
            assignments = []
            for dataset in distributed_datasets:
                local_assignments = self.assign_to_clusters(dataset, centroids)
                assignments.append(local_assignments)
                
            # Step 3: Securely compute new centroids
            new_centroids = self.secure_update_centroids(
                distributed_datasets, assignments
            )
            
            # Step 4: Check convergence
            if self.has_converged(centroids, new_centroids):
                break
                
            centroids = new_centroids
            
        return centroids, assignments
```

### Federated Learning vs SMC

**Federated Learning:**
- Participants train local models and share model updates
- Central server aggregates updates to create global model
- Provides some privacy but vulnerable to inference attacks

**SMC for Machine Learning:**
- No model updates are shared in the clear
- All computations done under encryption or secret sharing
- Stronger privacy guarantees but higher computational cost

**Comparison:**

| Aspect | Federated Learning | SMC-based ML |
|--------|-------------------|--------------|
| **Privacy** | Model updates visible | All computation private |
| **Efficiency** | High | Lower (crypto overhead) |
| **Communication** | Model-size dependent | Data-size dependent |
| **Trust** | Requires trusted aggregator | Fully distributed |
| **Inference Attacks** | Vulnerable | Resistant |

---

## 🔍 Detailed Explanations

### Challenges in SMC for Machine Learning

**1. Non-linear Operations:**
- **Problem:** Activation functions (ReLU, sigmoid, tanh) are expensive in SMC
- **Solution:** Polynomial approximations or garbled circuits

**2. Floating-Point Arithmetic:**
- **Problem:** SMC typically works with integers
- **Solution:** Fixed-point arithmetic with scaling factors

**3. Iterative Algorithms:**
- **Problem:** Many ML algorithms require iterations with communication overhead
- **Solution:** Reduce iterations, batch operations, optimize communication

**4. Model Size and Complexity:**
- **Problem:** Large neural networks have millions of parameters
- **Solution:** Model compression, layer-wise computation, pruning

### Privacy-Preserving Neural Networks

**Solutions for Deep Learning:**

**1. Linear Approximations:**
```python
def secure_relu_approximation(x):
    # Approximate ReLU with polynomial
    # ReLU(x) ≈ 0.5x + 0.5|x| ≈ polynomial approximation
    return secure_polynomial_evaluation(x, relu_coefficients)
```

**2. Secret Sharing for Linear Operations:**
```python
class SecretSharedLayer:
    def __init__(self, weights, bias):
        self.weights = secret_share_matrix(weights)
        self.bias = secret_share_vector(bias)
        
    def forward(self, input_shares):
        # Matrix multiplication under secret sharing
        output_shares = secure_matrix_multiply(input_shares, self.weights)
        output_shares = secure_vector_add(output_shares, self.bias)
        return output_shares
```

**3. Hybrid Approaches:**
```python
def hybrid_nn_protocol(layers):
    for layer in layers:
        if layer.type == "linear":
            result = secret_sharing_compute(layer)
        elif layer.type == "activation":
            result = garbled_circuits_compute(layer)
        elif layer.type == "pooling":
            result = homomorphic_compute(layer)
    return result
```

---

## 💡 Practical Examples

### Example 1: Private Medical Diagnosis

**Scenario:** Three hospitals want to build a joint diabetes prediction model

```python
class PrivateMedicalDiagnosis:
    def __init__(self, hospitals):
        self.hospitals = hospitals
        self.model = None
        
    def collaborative_training(self):
        # Step 1: Agree on feature set and preprocessing
        common_features = self.negotiate_features()
        
        # Step 2: Locally preprocess data
        preprocessed_data = []
        for hospital in self.hospitals:
            local_data = hospital.preprocess_data(common_features)
            preprocessed_data.append(local_data)
            
        # Step 3: Securely train logistic regression
        self.model = self.secure_logistic_regression(preprocessed_data)
        
        return self.model
        
    def secure_prediction(self, patient_features):
        # Each hospital can use the model for local predictions
        # without revealing individual patient data
        encrypted_features = encrypt_features(patient_features)
        encrypted_prediction = self.model.predict(encrypted_features)
        return decrypt_prediction(encrypted_prediction)
```

### Example 2: Private Market Basket Analysis

**Scenario:** Competing retailers want to find common purchasing patterns

```python
class PrivateMarketBasketAnalysis:
    def __init__(self, retailers, min_support=0.1):
        self.retailers = retailers
        self.min_support = min_support
        
    def find_association_rules(self):
        # Step 1: Standardize product catalogs
        unified_catalog = self.create_unified_catalog()
        
        # Step 2: Convert transactions to unified format
        standardized_transactions = []
        for retailer in self.retailers:
            transactions = retailer.get_transactions(unified_catalog)
            standardized_transactions.append(transactions)
            
        # Step 3: Securely mine frequent itemsets
        frequent_itemsets = self.secure_apriori(standardized_transactions)
        
        # Step 4: Generate association rules
        rules = self.generate_rules(frequent_itemsets)
        
        return rules
```

### Example 3: Private Credit Scoring

**Scenario:** Banks collaborate on fraud detection without sharing customer data

```python
class PrivateCreditScoring:
    def __init__(self, banks):
        self.banks = banks
        self.fraud_model = None
        
    def train_fraud_detection_model(self):
        # Step 1: Standardize feature representations
        feature_schema = self.agree_on_features()
        
        # Step 2: Each bank prepares local data
        local_datasets = []
        for bank in self.banks:
            local_data = bank.prepare_training_data(feature_schema)
            local_datasets.append(local_data)
            
        # Step 3: Securely train ensemble model
        self.fraud_model = self.secure_ensemble_training(local_datasets)
        
        return self.fraud_model
        
    def secure_prediction(self, transaction_features):
        # Encrypt transaction features
        encrypted_features = encrypt_transaction(transaction_features)
        
        # Each base model makes encrypted prediction
        encrypted_predictions = []
        for model in self.fraud_model.base_models:
            pred = model.secure_predict(encrypted_features)
            encrypted_predictions.append(pred)
            
        # Securely aggregate predictions (majority vote)
        final_prediction = self.secure_majority_vote(encrypted_predictions)
        
        return decrypt_prediction(final_prediction)
```

---

## ❓ Self-Assessment Questions

<details>
<summary><b>Question 1:</b> What are the main challenges in applying SMC to machine learning algorithms? How do these differ from traditional centralized ML? (Click to reveal answer)</summary>

**Answer:** 

**Main SMC-ML Challenges:**

**1. Non-linear Operations:**
- **Problem:** Activation functions (ReLU, sigmoid, tanh) are expensive in SMC
- **Traditional ML:** Direct computation of non-linear functions
- **SMC Solution:** Polynomial approximations, garbled circuits, or piecewise linear approximations

**2. Floating-Point Arithmetic:**
- **Problem:** SMC typically works with integers; floating-point is complex
- **Traditional ML:** Native floating-point support
- **SMC Solution:** Fixed-point arithmetic with scaling factors

**3. Iterative Algorithms:**
- **Problem:** Many ML algorithms require many iterations with communication overhead
- **Traditional ML:** Fast local iterations
- **SMC Solution:** Reduce iterations, batch operations, optimize communication

**4. Gradient Computation:**
- **Problem:** Backpropagation requires division and complex derivatives
- **Traditional ML:** Automatic differentiation
- **SMC Solution:** Approximate gradients, secure division protocols

**5. Model Size and Complexity:**
- **Problem:** Large neural networks have millions of parameters
- **Traditional ML:** Handle arbitrary model sizes
- **SMC Solution:** Model compression, layer-wise computation, pruning

**6. Performance Trade-offs:**
- **Problem:** 100-1000x slower than traditional ML
- **Traditional ML:** Optimized for speed
- **SMC Solution:** Hybrid approaches, precomputation, specialized hardware

**Key Differences:**
- **Computation:** Cryptographic operations vs. direct arithmetic
- **Communication:** Constant interaction vs. no communication
- **Precision:** Limited precision vs. full floating-point
- **Scalability:** Limited by crypto overhead vs. data/model size
- **Privacy:** Strong guarantees vs. no privacy protection

</details>

<details>
<summary><b>Question 2:</b> Compare the privacy guarantees and efficiency trade-offs between federated learning and SMC-based machine learning. (Click to reveal answer)</summary>

**Answer:** 

**Privacy Guarantees Comparison:**

**Federated Learning:**
- **What's Protected:** Raw data never leaves local devices
- **What's Exposed:** Model updates (gradients, weights) are shared
- **Attack Vulnerability:** 
  - Gradient inversion attacks can reconstruct training data
  - Model inversion attacks can extract sensitive features
  - Membership inference attacks can determine if specific data was used
- **Privacy Level:** Moderate (better than centralized, worse than SMC)

**SMC-based ML:**
- **What's Protected:** All intermediate computations and model updates
- **What's Exposed:** Only final agreed-upon results
- **Attack Resistance:** 
  - Strong protection against gradient-based attacks
  - Prevents model inversion attacks
  - Resistant to membership inference
- **Privacy Level:** High (cryptographic guarantees)

**Efficiency Comparison:**

| Metric | Federated Learning | SMC-based ML |
|--------|-------------------|--------------|
| **Computation** | ~1.5-3x overhead | ~100-1000x overhead |
| **Communication** | Model size per round | Depends on data/computation |
| **Rounds** | 10-100 rounds typical | Fewer rounds but heavier |
| **Scalability** | 100s-1000s participants | Typically <10 participants |
| **Training Time** | Hours to days | Days to weeks |

**When to Choose Each:**
- **Federated Learning:** Large-scale deployment, moderate privacy needs, performance critical
- **SMC-based ML:** High-value data, strong privacy requirements, small number of parties
- **Hybrid:** Balance privacy and efficiency for different computation phases

</details>

<details>
<summary><b>Question 3:</b> Design a secure protocol for collaborative outlier detection across multiple datasets. What are the key privacy and utility considerations? (Click to reveal answer)</summary>

**Answer:** 

**Secure Collaborative Outlier Detection Protocol:**

**Problem Setup:**
- Multiple organizations have datasets with similar structure
- Want to detect outliers that appear anomalous across all datasets
- Each organization's data must remain private
- Goal: Identify global outliers without revealing local patterns

**Protocol Design:**

**Phase 1: Secure Statistic Computation**
```python
class SecureOutlierDetection:
    def __init__(self, participants, threshold=2.0):
        self.participants = participants
        self.threshold = threshold  # Standard deviations for outlier
        
    def compute_global_statistics(self):
        # Step 1: Each party computes local statistics
        local_stats = []
        for participant in self.participants:
            stats = participant.compute_local_stats()  # mean, variance, count
            local_stats.append(stats)
            
        # Step 2: Securely aggregate statistics
        total_count = secure_sum([s['count'] for s in local_stats])
        
        # Secure computation of global mean
        weighted_means = []
        for stats in local_stats:
            weighted_mean = secure_multiply(stats['mean'], stats['count'])
            weighted_means.append(weighted_mean)
        global_mean = secure_divide(secure_sum(weighted_means), total_count)
        
        return global_mean, global_variance
```

**Privacy Considerations:**
1. **Statistical Privacy:** Add noise to protect individual contributions
2. **Pattern Privacy:** Group similar outliers to prevent identification
3. **Individual vs. Global Privacy:** Different guarantees for records vs. organizations

**Utility Considerations:**
1. **Detection Accuracy:** Balance privacy noise with detection effectiveness
2. **Statistical Utility:** Preserve meaningful statistical properties
3. **Real-time vs. Batch:** Different latency and privacy requirements

</details>

<details>
<summary><b>Question 4:</b> How would you handle the challenge of training deep neural networks using SMC? What are the bottlenecks and potential solutions? (Click to reveal answer)</summary>

**Answer:** 

**Deep Neural Network SMC Challenges:**

**1. Scale and Complexity:**
- **Problem:** Modern DNNs have millions/billions of parameters
- **SMC Impact:** Each parameter requires secure computation
- **Bottleneck:** Memory and computation grow linearly with model size

**2. Non-linear Activations:**
- **Problem:** ReLU, sigmoid, tanh are expensive in SMC
- **Traditional Cost:** O(1) per activation
- **SMC Cost:** O(log n) for comparison-based or O(n) for polynomial approximation

**Solutions and Optimizations:**

**1. Activation Function Approximations:**
```python
def secure_relu_polynomial(self, x):
    # Approximate ReLU with degree-3 polynomial
    abs_x = self.secure_absolute_value_approx(x)
    return self.secure_multiply(0.5, self.secure_add(x, abs_x))
```

**2. Hybrid Computation Approaches:**
```python
class HybridSecureDNN:
    def __init__(self, public_layers, private_layers):
        self.public_layers = public_layers    # Can be computed in clear
        self.private_layers = private_layers  # Must be computed securely
        
    def hybrid_training(self, data_shares):
        # Phase 1: Public feature extraction
        public_features = self.extract_public_features(data_shares)
        
        # Phase 2: Secure private computation
        private_features = self.secure_private_layers(public_features)
        
        # Phase 3: Public classification head
        final_output = self.public_classification(private_features)
        
        return final_output
```

**3. Model Architecture Optimizations:**
- Minimize non-linear activations
- Use wider layers instead of deeper networks
- Avoid normalization layers when possible
- Knowledge distillation from larger teacher models

**Current Limitations:**
- Deep networks with >1M parameters are impractical with current SMC
- Training time scales poorly with model complexity
- Limited to simple architectures and approximations

</details>

<details>
<summary><b>Question 5:</b> Explain the relationship between federated learning and differential privacy. How can they be combined for stronger privacy guarantees? (Click to reveal answer)</summary>

**Answer:** 

**Federated Learning Privacy Vulnerabilities:**
- **Gradient Leakage:** Model updates can leak information about training data
- **Inference Attacks:** Adversaries can infer membership, attributes, or even reconstruct data
- **Honest-but-Curious Servers:** Central aggregator sees all model updates

**How Differential Privacy Strengthens Federated Learning:**

**1. Local Differential Privacy (LDP) in FL:**
```python
class DPFederatedLearning:
    def __init__(self, epsilon_local, epsilon_global):
        self.epsilon_local = epsilon_local    # Local DP budget per client
        self.epsilon_global = epsilon_global  # Global DP budget for aggregation
        
    def client_update_with_ldp(self, local_model, local_data, learning_rate):
        # Standard gradient computation
        gradients = self.compute_gradients(local_model, local_data)
        
        # Add noise to gradients (Local DP)
        noisy_gradients = []
        for grad in gradients:
            sensitivity = self.compute_gradient_sensitivity(grad)
            noise = self.laplace_noise(sensitivity / self.epsilon_local)
            noisy_gradients.append(grad + noise)
            
        return updated_model, noisy_gradients
```

**2. Advanced Composition:**
- **Moments Accountant:** Provides tighter bounds than basic composition
- **Rényi Differential Privacy:** Better composition for iterative algorithms
- **Adaptive Noise Scheduling:** Allocate privacy budget optimally across rounds

**3. Client-Level Differential Privacy:**
```python
class ClientLevelDP:
    def __init__(self, epsilon_per_client, max_participation):
        self.epsilon_per_client = epsilon_per_client
        self.max_participation = max_participation
        
    def allocate_privacy_budget(self, client_id, round_number):
        remaining_rounds = self.max_participation - self.participation_counts[client_id]
        round_epsilon = self.epsilon_per_client / remaining_rounds
        return round_epsilon
```

**Benefits of DP-FL Combination:**
- **Formal Privacy Guarantees:** Mathematical bounds on privacy loss
- **Defense Against Inference Attacks:** Protection beyond just gradient hiding
- **Flexible Privacy-Utility Trade-offs:** Tunable parameters for different needs
- **Composition Properties:** Can reason about privacy across multiple training runs

**Challenges:**
- **Utility Degradation:** Additional noise reduces model accuracy
- **Complex Parameter Tuning:** Many hyperparameters to optimize
- **Implementation Complexity:** Requires expertise in both FL and DP

</details>

---

## 🔬 Lab Exercises

### Exercise 1: Private Classification Implementation

**Task:** Implement a privacy-preserving Naive Bayes classifier

```python
import numpy as np
from collections import defaultdict

class PrivateNaiveBayes:
    def __init__(self, num_parties, epsilon=1.0):
        self.num_parties = num_parties
        self.epsilon = epsilon
        self.class_priors = {}
        self.feature_likelihoods = {}
        
    def distributed_training(self, datasets):
        """
        Train Naive Bayes on distributed datasets
        datasets: list of (X, y) tuples, one per party
        """
        # Step 1: Compute local statistics
        local_stats = []
        for X, y in datasets:
            stats = self.compute_local_statistics(X, y)
            local_stats.append(stats)
            
        # Step 2: Aggregate statistics securely
        global_stats = self.secure_aggregation(local_stats)
        
        # Step 3: Compute final probabilities
        self.class_priors = self.compute_class_priors(global_stats)
        self.feature_likelihoods = self.compute_feature_likelihoods(global_stats)
        
        return self
        
    def compute_local_statistics(self, X, y):
        """Compute local counts for each party"""
        stats = {
            'class_counts': defaultdict(int),
            'feature_counts': defaultdict(lambda: defaultdict(int)),
            'total_samples': len(y)
        }
        
        for i, label in enumerate(y):
            stats['class_counts'][label] += 1
            
            for j, feature_value in enumerate(X[i]):
                stats['feature_counts'][j][label] += feature_value
                
        return stats
        
    def secure_aggregation(self, local_stats):
        """Simulate secure aggregation of local statistics"""
        global_stats = {
            'class_counts': defaultdict(int),
            'feature_counts': defaultdict(lambda: defaultdict(int)),
            'total_samples': 0
        }
        
        for stats in local_stats:
            # Aggregate class counts
            for class_label, count in stats['class_counts'].items():
                # Add differential privacy noise
                noisy_count = count + np.random.laplace(0, 1/self.epsilon)
                global_stats['class_counts'][class_label] += max(0, noisy_count)
                
            # Aggregate feature counts
            for feature_idx, feature_stats in stats['feature_counts'].items():
                for class_label, count in feature_stats.items():
                    noisy_count = count + np.random.laplace(0, 1/self.epsilon)
                    global_stats['feature_counts'][feature_idx][class_label] += max(0, noisy_count)
                    
            global_stats['total_samples'] += stats['total_samples']
            
        return global_stats
        
    def predict(self, X):
        """Make predictions on new data"""
        predictions = []
        
        for sample in X:
            class_scores = {}
            
            for class_label in self.class_priors:
                # Start with class prior
                score = np.log(self.class_priors[class_label])
                
                # Add feature likelihoods
                for feature_idx, feature_value in enumerate(sample):
                    if feature_idx in self.feature_likelihoods:
                        if class_label in self.feature_likelihoods[feature_idx]:
                            likelihood = self.feature_likelihoods[feature_idx][class_label]
                            if feature_value == 1:
                                score += np.log(likelihood + 1e-10)
                            else:
                                score += np.log(1 - likelihood + 1e-10)
                                
                class_scores[class_label] = score
                
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)
            
        return predictions

# Test implementation
def generate_synthetic_data(n_samples, n_features, n_classes):
    X = np.random.binomial(1, 0.3, (n_samples, n_features))
    y = np.random.randint(0, n_classes, n_samples)
    return X, y

# Create datasets for 3 parties
party_datasets = []
for i in range(3):
    X, y = generate_synthetic_data(100, 10, 2)
    party_datasets.append((X, y))

# Train private Naive Bayes
classifier = PrivateNaiveBayes(num_parties=3, epsilon=1.0)
classifier.distributed_training(party_datasets)

# Test prediction
test_X, test_y = generate_synthetic_data(20, 10, 2)
predictions = classifier.predict(test_X)
accuracy = np.mean(predictions == test_y)

print(f"Classification accuracy: {accuracy:.3f}")
print(f"Privacy parameter epsilon: {classifier.epsilon}")
```

### Exercise 2: Private K-Means Clustering

**Task:** Implement privacy-preserving k-means clustering

```python
class PrivateKMeans:
    def __init__(self, k, max_iterations=10, epsilon=1.0):
        self.k = k
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.centroids = None
        
    def fit(self, distributed_datasets):
        """Perform k-means clustering on distributed data"""
        # Initialize centroids randomly
        self.centroids = self.initialize_centroids(distributed_datasets)
        
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}")
            
            # Step 1: Assign points to clusters (done locally)
            assignments = []
            for dataset in distributed_datasets:
                local_assignments = self.assign_to_clusters(dataset)
                assignments.append(local_assignments)
                
            # Step 2: Update centroids securely
            new_centroids = self.secure_update_centroids(distributed_datasets, assignments)
            
            # Step 3: Check convergence
            if self.has_converged(new_centroids):
                print("Converged!")
                break
                
            self.centroids = new_centroids
            
        return self
        
    def initialize_centroids(self, distributed_datasets):
        """Initialize centroids using data from all parties"""
        all_points = []
        for dataset in distributed_datasets:
            all_points.extend(dataset)
            
        # Randomly select k initial centroids
        import random
        centroids = random.sample(all_points, self.k)
        return np.array(centroids)
        
    def assign_to_clusters(self, dataset):
        """Assign each point to nearest centroid"""
        assignments = []
        for point in dataset:
            distances = [np.linalg.norm(point - centroid) for centroid in self.centroids]
            closest_cluster = np.argmin(distances)
            assignments.append(closest_cluster)
        return assignments
        
    def secure_update_centroids(self, datasets, assignments):
        """Securely compute new centroids"""
        new_centroids = []
        
        for cluster_id in range(self.k):
            # Collect points assigned to this cluster from all parties
            cluster_points = []
            
            for party_idx, (dataset, party_assignments) in enumerate(zip(datasets, assignments)):
                for point_idx, assignment in enumerate(party_assignments):
                    if assignment == cluster_id:
                        cluster_points.append(dataset[point_idx])
                        
            if cluster_points:
                # Compute centroid with differential privacy
                cluster_points = np.array(cluster_points)
                centroid = np.mean(cluster_points, axis=0)
                
                # Add noise for differential privacy
                sensitivity = self.compute_sensitivity(datasets)
                noise = np.random.laplace(0, sensitivity/self.epsilon, centroid.shape)
                noisy_centroid = centroid + noise
                
                new_centroids.append(noisy_centroid)
            else:
                # Keep old centroid if no points assigned
                new_centroids.append(self.centroids[cluster_id])
                
        return np.array(new_centroids)
        
    def compute_sensitivity(self, datasets):
        """Compute sensitivity for differential privacy"""
        all_points = []
        for dataset in datasets:
            all_points.extend(dataset)
            
        max_distance = 0
        for point in all_points:
            for centroid in self.centroids:
                distance = np.linalg.norm(point - centroid)
                max_distance = max(max_distance, distance)
                
        return max_distance / len(all_points)
        
    def has_converged(self, new_centroids, tolerance=1e-4):
        """Check if centroids have converged"""
        if self.centroids is None:
            return False
            
        distance = np.linalg.norm(self.centroids - new_centroids)
        return distance < tolerance

# Test implementation
def generate_clustered_data(n_samples, n_features, n_clusters):
    """Generate data with natural clusters"""
    data = []
    for _ in range(n_clusters):
        center = np.random.uniform(-10, 10, n_features)
        cluster_points = np.random.normal(center, 2, (n_samples//n_clusters, n_features))
        data.extend(cluster_points)
    return np.array(data)

# Create datasets for 3 parties
party_datasets = []
for i in range(3):
    data = generate_clustered_data(150, 2, 3)
    party_datasets.append(data)

# Perform private k-means clustering
kmeans = PrivateKMeans(k=3, max_iterations=10, epsilon=1.0)
kmeans.fit(party_datasets)

print("Final centroids:")
for i, centroid in enumerate(kmeans.centroids):
    print(f"Cluster {i}: {centroid}")
```

### Exercise 3: Privacy-Preserving Model Evaluation

**Task:** Implement secure model evaluation across multiple parties

```python
class PrivateModelEvaluation:
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
        
    def secure_accuracy_computation(self, predictions_list, labels_list):
        """Compute accuracy across multiple parties without revealing individual results"""
        total_correct = 0
        total_samples = 0
        
        for predictions, labels in zip(predictions_list, labels_list):
            # Each party computes local accuracy statistics
            local_correct = np.sum(predictions == labels)
            local_total = len(labels)
            
            # Add differential privacy noise
            noisy_correct = local_correct + np.random.laplace(0, 1/self.epsilon)
            noisy_total = local_total + np.random.laplace(0, 1/self.epsilon)
            
            # Ensure non-negative values
            noisy_correct = max(0, noisy_correct)
            noisy_total = max(1, noisy_total)
            
            total