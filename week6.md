# Week 6: Secure Multiparty Computation (SMC)

<div align="center">

[⬅️ **Week 5**](week5.md) | [**Main**](README.md) | [**Week 7** ➡️](week7.md)

</div>

---

## 🎯 Learning Goals

By the end of this week, you should understand:
- The fundamentals of Secure Multiparty Computation (SMC)
- Key SMC protocols: Yao's Garbled Circuits, BGW, GMW
- Privacy guarantees and security models in SMC
- Practical applications and limitations of SMC systems

---

## 📖 Theoretical Content

### Introduction to Secure Multiparty Computation

**Definition:** Secure Multiparty Computation enables multiple parties to jointly compute a function over their inputs while keeping those inputs private.

**Classical Example - Yao's Millionaire Problem:**
Two millionaires want to determine who is richer without revealing their exact wealth.
- Alice has wealth A, Bob has wealth B
- Goal: Compute A > B without revealing A or B
- Neither party learns anything beyond the comparison result

**Formal Requirements:**
1. **Correctness:** Output equals the result of the ideal computation
2. **Privacy:** No party learns more than what can be inferred from their input and output
3. **Independence of Inputs:** Parties cannot choose inputs based on others' inputs
4. **Guaranteed Output Delivery:** Honest parties receive correct output
5. **Fairness:** Either all parties get output or none do

### Security Models

**1. Semi-Honest (Honest-but-Curious) Model**
- Parties follow the protocol correctly
- But may try to learn extra information from messages
- Easier to achieve, many practical protocols
- Suitable when participants have incentives to cooperate

**2. Malicious (Byzantine) Model**
- Parties may deviate arbitrarily from protocol
- May provide false inputs or corrupt messages
- Stronger security guarantees
- More complex and expensive protocols

**3. Covert Model**
- Intermediate between semi-honest and malicious
- Cheating is detected with some probability
- Practical middle ground for many applications

### Threshold Security

**t-out-of-n Security:**
- Protocol secure as long as at most t parties are corrupted
- Common thresholds:
  - **Honest Majority:** t < n/2 (most efficient)
  - **Dishonest Majority:** t < n (requires stronger assumptions)

### Circuit Representation

**Boolean Circuits:**
Functions represented as networks of AND, OR, and NOT gates
- **Addition:** Full adder circuits
- **Comparison:** Comparator circuits  
- **Multiplication:** More complex, quadratic circuit size

**Arithmetic Circuits:**
Functions over finite fields or rings
- **Addition:** Simple field addition
- **Multiplication:** More expensive operation
- **Suitable for numerical computations**

---

## 🔍 Detailed Explanations

### Yao's Garbled Circuits Protocol

**High-Level Idea:**
1. **Garbler** (Alice) creates "garbled" version of circuit
2. **Evaluator** (Bob) evaluates garbled circuit
3. Result revealed without exposing intermediate values

**Garbling Process:**
1. **Wire Labels:** Each wire gets two random labels (one for 0, one for 1)
2. **Gate Tables:** For each gate, create encrypted truth table
3. **Encryption:** Use wire labels as keys for encryption

**Example - AND Gate:**
```
Truth Table:     Garbled Table (encrypted):
a | b | out      Enc(k_a0, k_b0, k_out0)  [for 0 AND 0 = 0]
0 | 0 | 0        Enc(k_a0, k_b1, k_out0)  [for 0 AND 1 = 0]
0 | 1 | 0        Enc(k_a1, k_b0, k_out0)  [for 1 AND 0 = 0]
1 | 0 | 0        Enc(k_a1, k_b1, k_out1)  [for 1 AND 1 = 1]
1 | 1 | 1
```

**Evaluation Process:**
1. Bob receives garbled circuit from Alice
2. Bob obtains wire labels for his input via Oblivious Transfer
3. Bob evaluates circuit gate by gate
4. Final output wire labels reveal computation result

### Secret Sharing

**Shamir's Secret Sharing:**
Split secret s into n shares such that any t shares can reconstruct s

**Polynomial Construction:**
- Choose random polynomial f(x) of degree t-1
- Set f(0) = s (the secret)
- Shares are f(1), f(2), ..., f(n)
- Any t points determine unique polynomial of degree t-1

**BGW Protocol (Ben-Or, Goldwasser, Wigderson):**
- Based on secret sharing over finite fields
- Addition: Add corresponding shares
- Multiplication: More complex, requires interaction
- Secure against t < n/3 malicious adversaries

### GMW Protocol (Goldreich, Micali, Wigderson)

**Share-based approach for Boolean circuits:**
1. **Secret sharing:** Use XOR-based sharing (s = s₁ ⊕ s₂ ⊕ ... ⊕ sₙ)
2. **XOR gates:** Free computation (XOR shares)
3. **AND gates:** Require secure multiplication protocol
4. **Communication:** Interaction needed for AND gates only

**Advantages:**
- Simple share structure (XOR-based)
- Efficient for circuits with many XOR operations
- Generalizes to any number of parties
- Secure against t < n/2 semi-honest adversaries

**Disadvantages:**
- Communication-intensive for AND-heavy circuits
- Round complexity proportional to circuit depth

---

## 💡 Practical Examples

### Example 1: Private Set Intersection (PSI)

**Scenario:** Two companies want to find common customers without revealing their entire customer lists.

**Setup:**
- Company A has customer set S_A = {Alice, Bob, Charlie}
- Company B has customer set S_B = {Bob, Charlie, David}
- Goal: Find S_A ∩ S_B = {Bob, Charlie}

**SMC Solution Using Circuit:**
```python
def psi_circuit(set_a, set_b):
    intersection = []
    for item_a in set_a:
        for item_b in set_b:
            if secure_equal(item_a, item_b):  # SMC equality test
                intersection.append(item_a)
    return intersection
```

**Privacy Guarantees:**
- Neither company learns about customers not in intersection
- Only mutual customers are revealed
- No information about set sizes (with proper padding)

### Example 2: Private Bidding Auction

**Scenario:** Sealed-bid auction where highest bidder wins, but losing bids remain private.

**Participants:**
- Bidder 1: $100
- Bidder 2: $150  
- Bidder 3: $120

**SMC Protocol:**
1. **Input Phase:** Each bidder secret-shares their bid
2. **Computation Phase:** Compute maximum using comparison circuits
3. **Output Phase:** Reveal winner and winning bid only

**Comparison Circuit for Two Values:**
```
Function: max(a, b)
1. Compute comparison: c = (a > b)
2. If c = 1, return a; else return b
3. Circuit depth: O(log(bit_length))
```

**Benefits:**
- Losing bidders' privacy protected
- Prevents bid manipulation based on others' bids
- Ensures fair auction process

### Example 3: Private Statistical Analysis

**Scenario:** Multiple hospitals want to compute average patient age for research without sharing individual patient data.

**Data:**
- Hospital A: Patients aged [25, 30, 35, 40]
- Hospital B: Patients aged [28, 33, 38]  
- Hospital C: Patients aged [22, 27, 32, 37, 42]

**SMC Computation:**
```python
def secure_average(datasets):
    # Step 1: Compute total sum
    total_sum = 0
    for dataset in datasets:
        for age in dataset:
            total_sum += secure_add(age)  # SMC addition
    
    # Step 2: Compute total count
    total_count = 0
    for dataset in datasets:
        total_count += secure_add(len(dataset))
    
    # Step 3: Secure division
    return secure_divide(total_sum, total_count)
```

**Result:** Average age ≈ 32.1 years (computed securely)

**Privacy Benefits:**
- Individual patient ages never revealed
- Hospital-specific statistics remain private
- Only aggregate result is disclosed

---

## ❓ Self-Assessment Questions

<details>
<summary><b>Question 1:</b> Explain the difference between semi-honest and malicious security models in SMC. When would you use each? (Click to reveal answer)</summary>

**Answer:** 

**Semi-Honest (Honest-but-Curious) Model:**
- **Behavior:** Parties follow protocol correctly but try to learn extra information
- **Assumptions:** Participants have incentive to cooperate (e.g., mutual benefit)
- **Security:** Protects against passive eavesdropping on protocol execution
- **Efficiency:** Generally more efficient protocols
- **Use Cases:** Research collaborations, industry consortiums with aligned interests

**Malicious (Byzantine) Model:**
- **Behavior:** Parties may deviate arbitrarily from protocol
- **Assumptions:** No trust between participants, potential adversarial behavior
- **Security:** Protects against active attacks, input manipulation, protocol corruption
- **Efficiency:** More complex and expensive protocols
- **Use Cases:** Financial transactions, legal disputes, competitive business scenarios

**When to Use Each:**

**Semi-Honest Appropriate:**
- Medical research collaboration between hospitals
- Academic institutions sharing data for research
- Companies in non-competitive alliance
- Situations with legal/contractual enforcement

**Malicious Required:**
- Financial auctions and trading
- Voting systems
- Competitive business intelligence
- Legal proceedings requiring high assurance
- Cryptocurrency and blockchain applications

**Practical Consideration:** Many real-world systems use semi-honest protocols with additional mechanisms (reputation, auditing, legal agreements) to discourage malicious behavior.

</details>

<details>
<summary><b>Question 2:</b> In Yao's Garbled Circuits, why can't the evaluator learn anything about the garbler's input from the garbled circuit? (Click to reveal answer)</summary>

**Answer:** 

**Cryptographic Protection Mechanisms:**

**1. Random Wire Labels:**
- Each wire has two random labels (for 0 and 1)
- Labels are computationally indistinguishable from random
- Evaluator cannot determine which label corresponds to which bit value

**2. Encrypted Gate Tables:**
- Truth table entries are encrypted using wire labels as keys
- Evaluator can only decrypt one entry per gate (corresponding to actual computation path)
- Cannot decrypt other entries to learn alternative input combinations

**3. Point-and-Permute Optimization:**
- Wire labels include pointer bits indicating correct ciphertext
- Prevents evaluator from trying all decryptions
- Maintains security while improving efficiency

**4. Oblivious Transfer for Input:**
- Evaluator receives only wire labels corresponding to their actual input bits
- Cannot obtain labels for other possible inputs
- Garbler doesn't learn evaluator's input choice

**Example:**
```
Garbler's input: a = 1
Evaluator's input: b = 0

Evaluator receives:
- Label for a=1 (but doesn't know it represents 1)
- Label for b=0 (corresponding to their input)
- Can only decrypt gate table entry for (1,0) combination
- Cannot decrypt entries for (0,0), (0,1), or (1,1)
```

**Information-Theoretic Perspective:**
From evaluator's view, the garbled circuit is consistent with any possible input from the garbler, providing perfect privacy for the garbler's input.

</details>

<details>
<summary><b>Question 3:</b> Design an SMC protocol for the following scenario: Three friends want to find out who should pay for dinner (person with highest salary) without revealing their actual salaries. (Click to reveal answer)</summary>

**Answer:** 

**Scenario:** Alice, Bob, and Charlie need to determine who has the highest salary for dinner payment without revealing actual amounts.

**Protocol Design:**

**Step 1: Input Preparation**
```python
# Each person has salary: Alice (A), Bob (B), Charlie (C)
# Convert to binary representation (assume 32-bit salaries)
A_bits = binary_representation(alice_salary)
B_bits = binary_representation(bob_salary)  
C_bits = binary_representation(charlie_salary)
```

**Step 2: Secret Sharing (Using Shamir's Scheme)**
```python
# Each person secret-shares their salary
alice_shares = secret_share(alice_salary, threshold=2, parties=3)
bob_shares = secret_share(bob_salary, threshold=2, parties=3)
charlie_shares = secret_share(charlie_salary, threshold=2, parties=3)

# Distribute shares
# Alice gets: alice_shares[0], bob_shares[0], charlie_shares[0]
# Bob gets: alice_shares[1], bob_shares[1], charlie_shares[1]  
# Charlie gets: alice_shares[2], bob_shares[2], charlie_shares[2]
```

**Step 3: Secure Comparison Circuit**
```python
def secure_max_of_three(a_shares, b_shares, c_shares):
    # Compare A vs B
    ab_comparison = secure_comparison(a_shares, b_shares)  # 1 if A>B, 0 otherwise
    max_ab = secure_conditional_select(ab_comparison, a_shares, b_shares)
    winner_ab = secure_conditional_select(ab_comparison, "Alice", "Bob")
    
    # Compare max(A,B) vs C
    max_abc_comparison = secure_comparison(max_ab, c_shares)
    final_winner = secure_conditional_select(max_abc_comparison, winner_ab, "Charlie")
    
    return final_winner
```

**Step 4: Protocol Execution**
1. **Round 1:** Exchange secret shares
2. **Round 2:** Jointly evaluate comparison circuit
3. **Round 3:** Reconstruct only the winner's identity

**Alternative Approaches:**

**Approach 1: Yao's Garbled Circuits**
- Alice creates garbled circuit for 3-way comparison
- Bob and Charlie use Oblivious Transfer for inputs
- Output: Winner's name only

**Approach 2: Additive Secret Sharing**
```python
# Each salary split as: salary = share1 + share2 + share3
# Comparison done on shares without reconstruction

def compare_shared_values(shares_a, shares_b):
    # Compute shares_a - shares_b
    diff_shares = [a - b for a, b in zip(shares_a, shares_b)]
    # Secure sign test on difference
    return secure_sign_test(diff_shares)
```

**Privacy Properties:**
- No one learns actual salary amounts
- Only winner's identity is revealed
- Protocol secure against any single corrupt party
- Can be extended to handle ties (split payment)

**Optimizations:**
- Use approximation for efficiency (compare salary ranges)
- Batch multiple dinner decisions
- Add noise for differential privacy

</details>

<details>
<summary><b>Question 4:</b> What are the main performance bottlenecks in SMC protocols, and how do modern systems address them? (Click to reveal answer)</summary>

**Answer:** 

**Main Performance Bottlenecks:**

**1. Communication Complexity**
- **Problem:** SMC requires extensive message exchange between parties
- **Garbled Circuits:** O(|C|) communication where |C| is circuit size
- **Secret Sharing:** Multiplication gates require interaction rounds

**2. Computational Overhead**
- **Problem:** Cryptographic operations are expensive
- **Garbling:** Creating encrypted truth tables for each gate
- **Oblivious Transfer:** Expensive for input sharing

**3. Round Complexity**
- **Problem:** Sequential gates create communication rounds
- **Circuit Depth:** Deep circuits require many sequential rounds
- **Latency:** Network delays amplified by round count

**4. Circuit Size**
- **Problem:** Boolean circuits can be very large
- **Multiplication:** Quadratic blowup for arithmetic operations
- **Comparison:** Logarithmic depth but wide circuits

**Modern Solutions:**

**1. Communication Optimizations**
```python
# Preprocessing Phase (offline)
def offline_preprocessing():
    # Generate random multiplication triples
    # Create garbled circuits in advance
    # Prepare OT correlations
    pass

# Online Phase (fast)
def online_computation(inputs):
    # Use preprocessed materials
    # Minimal communication
    # Fast execution
    pass
```

**2. Algorithmic Improvements**
- **Free XOR:** XOR gates computed without communication
- **Half Gates:** Reduce garbled circuit size by 50%
- **Batching:** Amortize costs across multiple computations

**3. Hardware Acceleration**
```python
# GPU acceleration for parallel operations
class SMCAccelerator:
    def __init__(self):
        self.gpu_context = initialize_gpu()
    
    def parallel_garbling(self, circuit):
        # Garble multiple gates simultaneously
        return gpu_parallel_encrypt(circuit.gates)
    
    def batch_ot(self, choices):
        # Batch oblivious transfers
        return gpu_batch_ot(choices)
```

**4. Protocol-Specific Optimizations**

**Garbled Circuits:**
- **Pipeline:** Overlap garbling, transfer, and evaluation
- **Streaming:** Process circuit in chunks
- **Compression:** Reduce garbled table sizes

**Secret Sharing:**
- **Packed Secret Sharing:** Multiple secrets per sharing
- **BGV/CKKS:** Homomorphic encryption for arithmetic
- **Replicated Sharing:** Honest majority protocols

**5. Hybrid Approaches**
```python
def hybrid_computation(function):
    if is_arithmetic_heavy(function):
        return secret_sharing_protocol(function)
    elif is_boolean_heavy(function):
        return garbled_circuits_protocol(function)
    else:
        return mixed_protocol(function)
```

**Performance Metrics:**
- **Latency:** Sub-second for simple computations
- **Throughput:** Thousands of operations per second
- **Scalability:** Hundreds of parties in some protocols

**Real-World Performance:**
- **Simple comparisons:** ~1ms
- **Private set intersection:** ~seconds for thousands of items
- **Machine learning inference:** ~minutes for neural networks

</details>

<details>
<summary><b>Question 5:</b> Explain the trade-offs between different SMC approaches for implementing private machine learning. (Click to reveal answer)</summary>

**Answer:** 

**SMC Approaches for Private ML:**

**1. Garbled Circuits**
```python
# Neural network as boolean circuit
def nn_garbled_circuit(weights, inputs):
    # Convert floating point to fixed point
    # Implement each layer as circuit
    # ReLU activation as comparison + selection
    pass
```

**Advantages:**
- Constant rounds (low latency)
- Good for small, deep networks
- Handles non-linear activations well

**Disadvantages:**
- Expensive for large networks
- Fixed-point arithmetic limitations
- Circuit size grows with precision

**2. Homomorphic Encryption**
```python
# Encrypted computation on encrypted data
def nn_homomorphic(encrypted_weights, encrypted_inputs):
    # Linear operations: direct computation
    # Non-linear: approximation with polynomials
    return encrypted_prediction
```

**Advantages:**
- Minimal communication
- Good for linear operations (convolutions)
- Supports high precision

**Disadvantages:**
- Limited operations (mainly addition/multiplication)
- Polynomial approximations for activations
- High computational overhead

**3. Secret Sharing (SPDZ/ABY³)**
```python
# Arithmetic sharing for neural networks
def nn_secret_sharing(shared_weights, shared_inputs):
    # Matrix multiplication: efficient
    # ReLU: requires comparison protocol
    return shared_prediction
```

**Advantages:**
- Efficient arithmetic operations
- Scales to large networks
- Good throughput

**Disadvantages:**
- Many communication rounds
- Expensive non-linear operations
- Requires honest majority (some protocols)

**4. Hybrid Approaches**
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

**Trade-off Analysis:**

**Latency vs Throughput:**
- **Garbled Circuits:** Low latency, low throughput
- **Secret Sharing:** Higher latency, high throughput
- **Homomorphic:** Variable latency, medium throughput

**Network Size:**
- **Small networks (<1000 parameters):** Garbled circuits preferred
- **Medium networks (1K-1M parameters):** Hybrid approaches
- **Large networks (>1M parameters):** Secret sharing or specialized protocols

**Precision Requirements:**
- **Low precision (8-16 bits):** All approaches viable
- **High precision (32+ bits):** Homomorphic encryption or high-precision secret sharing

**Security Models:**
- **Semi-honest:** All approaches available
- **Malicious:** Limited options, typically secret sharing with verification

**Practical Considerations:**

**Training vs Inference:**
```python
# Training: requires backward pass
def private_training():
    # Forward pass + backward pass
    # Gradient computation and updates
    # More complex, typically secret sharing
    pass

# Inference: forward pass only  
def private_inference():
    # Single forward pass
    # Can use garbled circuits for low latency
    pass
```

**Real-World Performance:**
- **Image classification (ResNet-18):** ~1-10 seconds per image
- **Linear regression:** ~100ms per prediction
- **Deep neural networks:** Minutes for training epochs

**Emerging Trends:**
- **Specialized hardware:** Crypto-processors for SMC
- **Approximate computation:** Trade accuracy for speed
- **Federated learning integration:** Combine with distributed training

</details>

---

## 🔬 Lab Exercises

### Exercise 1: Implementing Yao's Millionaire Protocol

**Task:** Implement a simple version of Yao's millionaire problem

```python
def millionaire_problem(alice_wealth, bob_wealth):
    """
    Secure comparison of two values
    Returns: True if alice_wealth > bob_wealth, False otherwise
    """
    # Your implementation here
    pass

# Test cases
assert millionaire_problem(1000000, 500000) == True
assert millionaire_problem(750000, 1200000) == False
```

**Steps:**
1. Design comparison circuit
2. Implement garbling procedure
3. Simulate oblivious transfer
4. Evaluate garbled circuit

### Exercise 2: Secret Sharing Implementation

**Task:** Implement Shamir's secret sharing scheme

```python
class SecretSharing:
    def __init__(self, threshold, num_parties):
        self.t = threshold
        self.n = num_parties
        
    def share_secret(self, secret):
        """Generate n shares of secret"""
        pass
        
    def reconstruct_secret(self, shares):
        """Reconstruct secret from shares"""
        pass

# Test with threshold 3-out-of-5
ss = SecretSharing(3, 5)
shares = ss.share_secret(12345)
reconstructed = ss.reconstruct_secret(shares[:3])
assert reconstructed == 12345
```

### Exercise 3: Private Set Intersection

**Task:** Implement PSI for two small sets

```python
def private_set_intersection(set_a, set_b):
    """
    Find intersection without revealing non-intersecting elements
    """
    # Use SMC techniques
    pass

# Example
company_a_customers = {"alice", "bob", "charlie"}
company_b_customers = {"bob", "charlie", "david"}
intersection = private_set_intersection(company_a_customers, company_b_customers)
# Should return {"bob", "charlie"}
```

---

## 📚 Additional Resources

### Foundational Papers
- Yao, A. C. (1982). "Protocols for secure computations"
- Ben-Or, M. et al. (1988). "Completeness theorems for non-cryptographic fault-tolerant distributed computation"
- Goldreich, O. et al. (1987). "How to play any mental game"

### Modern Protocols
- Damgård, I. et al. (2012). "Multiparty computation from somewhat homomorphic encryption"
- Zahur, S. et al. (2015). "Two halves make a whole: Reducing data transfer in garbled circuits"

### Practical Systems
- **SCALE-MAMBA:** Large-scale SMC platform
- **ABY Framework:** Mixed protocol SMC
- **MP-SPDZ:** Versatile SMC compiler
- **EMP Toolkit:** Efficient secure computation

### Applications
- **Private machine learning:** CrypTFlow, SecureML
- **Private database queries:** PIR systems
- **Secure auctions:** Financial trading platforms

---

## 🚀 Project 1 Defense Preparation

**Defense Week - Key Points:**
- Demonstrate understanding of all anonymization methods
- Compare privacy-utility trade-offs clearly
- Show practical implementation results
- Discuss real-world deployment considerations
- Address questions about limitations and future work

**Common Questions:**
1. Which method would you recommend for your dataset and why?
2. How do your results compare to theoretical expectations?
3. What are the computational costs of different approaches?
4. How would you handle larger datasets?
5. What privacy guarantees can you provide?

---

<div align="center">

[⬅️ **Week 5**](week5.md) | [**Main**](README.md) | [**Week 7** ➡️](week7.md)

</div>