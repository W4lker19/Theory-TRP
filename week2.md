# Week 2: Privacy-preserving Data Publishing I

<div align="center">
   <div align="center">
  <a href="week1.md">⬅️ <strong>Week 1</strong></a> |
  <a href="README.md"><strong>Main</strong></a> |
  <a href="week2.md"><strong>Week 2</strong> ➡️</a>
</div>
   
---

## 🎯 Learning Goals

By the end of this week, you should understand:
- The fundamentals of data anonymization and its challenges
- k-anonymity: definition, implementation, and limitations
- Quasi-identifiers and their role in re-identification attacks
- The trade-off between privacy and data utility

---

## 📖 Theoretical Content

### Introduction to Privacy-Preserving Data Publishing

**Data Publishing Scenarios:**
- Medical research datasets
- Census and demographic data
- Location-based service data
- Social network datasets
- Financial transaction records

**The Challenge:** How to publish useful data while protecting individual privacy?

**Traditional Approach - Direct Identifiers Removal:**
Simply removing names, SSNs, and other obvious identifiers was thought sufficient. However, this naive approach fails because:
- Quasi-identifiers can be combined for re-identification
- Auxiliary information enables linkage attacks
- Statistical disclosure can reveal sensitive information

### Quasi-Identifiers (QIDs)

**Definition:** Attributes that, while not uniquely identifying on their own, can be combined to re-identify individuals when linked with external data.

**Common Quasi-Identifiers:**
- **Demographic:** Age, gender, race/ethnicity
- **Geographic:** ZIP code, city, state
- **Temporal:** Date of birth, dates of events
- **Behavioral:** Preferences, activity patterns

**The Linkage Attack Problem:**
1. Attacker obtains published anonymized dataset
2. Attacker has auxiliary information about target individual
3. Attacker uses QIDs to narrow down records
4. Target individual is re-identified

### k-Anonymity Model

**Definition:** A dataset satisfies k-anonymity if every record is indistinguishable from at least k-1 other records with respect to quasi-identifiers.

**Key Concepts:**
- **Equivalence Class:** Group of records with identical QID values
- **k-anonymous:** Each equivalence class contains at least k records
- **Generalization:** Making values less specific (age 25 → age group 20-30)
- **Suppression:** Removing certain values or records entirely

**Formal Definition:**
Let T be a table and QI be the set of quasi-identifiers. T satisfies k-anonymity if and only if each sequence of values in T[QI] appears at least k times in T[QI].

### Generalization and Suppression Techniques

**Generalization Hierarchies:**

1. **Age Generalization:**
   ```
   Specific Age → Age Range → Broader Range
   25 → 20-30 → 20-40
   ```

2. **Geographic Generalization:**
   ```
   Street Address → ZIP Code → City → State → Country
   123 Main St → 12345 → Boston → MA → USA
   ```

3. **Date Generalization:**
   ```
   Full Date → Month/Year → Year → Decade
   2024-03-15 → 03/2024 → 2024 → 2020s
   ```

**Suppression Strategies:**
- **Cell Suppression:** Replace specific values with '*'
- **Record Suppression:** Remove entire records
- **Attribute Suppression:** Remove entire columns

### Information Loss and Utility Metrics

**Measuring Information Loss:**
1. **Discernibility Metric (DM):** Sum of squares of equivalence class sizes
2. **Normalized Certainty Penalty (NCP):** Measures generalization level
3. **Loss Metric (LM):** Proportion of lost information

**Utility Preservation:**
- Maintain statistical properties
- Preserve data distributions
- Support intended analysis tasks
- Minimize false patterns

---

## 🔍 Detailed Explanations

### The k-Anonymity Algorithm Process

**Step 1: Identify Quasi-Identifiers**
- Analyze dataset attributes
- Consider potential auxiliary information
- Assess re-identification risk

**Step 2: Define Generalization Hierarchies**
- Create value hierarchies for each QID
- Balance specificity with privacy needs
- Consider domain knowledge

**Step 3: Apply k-Anonymity Algorithm**
- **Mondrian Algorithm:** Recursive partitioning approach
- **Incognito Algorithm:** Bottom-up lattice traversal
- **Datafly Algorithm:** Greedy generalization approach

**Step 4: Verify and Optimize**
- Check k-anonymity constraint satisfaction
- Minimize information loss
- Validate utility preservation

### Real-World Case Study: Netflix Prize Dataset

**Background:** Netflix released 100 million movie ratings for a privacy-preserving recommendation contest.

**Anonymization Approach:**
- Removed subscriber names and personal info
- Kept only movie ratings and dates
- Added some noise to ratings

**The Attack (Narayanan & Shmatikov, 2008):**
- Used IMDb public ratings as auxiliary information
- Correlated viewing patterns between datasets
- Successfully re-identified Netflix subscribers

**Lessons Learned:**
- Sparse datasets are particularly vulnerable
- Temporal patterns can be quasi-identifiers
- Background knowledge attacks are powerful
- Simple anonymization is insufficient

### Understanding Equivalence Classes

**Example Dataset - Medical Records:**

| Patient | Age | Gender | ZIP | Condition |
|---------|-----|--------|-----|-----------|
| A       | 25  | F      | 02141 | Diabetes  |
| B       | 25  | F      | 02141 | Heart Disease |
| C       | 27  | M      | 02142 | Diabetes  |
| D       | 27  | M      | 02142 | Flu       |

**After 2-Anonymity (Age Generalization):**

| Patient | Age | Gender | ZIP | Condition |
|---------|-----|--------|-----|-----------|
| A       | 20-30 | F    | 02141 | Diabetes  |
| B       | 20-30 | F    | 02141 | Heart Disease |
| C       | 20-30 | M    | 02142 | Diabetes  |
| D       | 20-30 | M    | 02142 | Flu       |

**Equivalence Classes:**
- {A, B}: (20-30, F, 02141)
- {C, D}: (20-30, M, 02142)

Each class has size ≥ 2, satisfying 2-anonymity.

---

## 💡 Practical Examples

### Example 1: Implementing k-Anonymity

**Scenario:** Anonymizing a hospital patient dataset

**Original Data:**
```
Age | Gender | ZIP   | Disease
23  | M      | 02139 | Asthma
24  | M      | 02139 | Diabetes
25  | F      | 02142 | Asthma
26  | F      | 02142 | Heart Disease
```

**Step 1:** QIDs = {Age, Gender, ZIP}

**Step 2:** Check k-anonymity (k=2)
- Record 1: (23, M, 02139) - appears 1 time ❌
- Record 2: (24, M, 02139) - appears 1 time ❌
- Record 3: (25, F, 02142) - appears 1 time ❌
- Record 4: (26, F, 02142) - appears 1 time ❌

**Step 3:** Apply generalization

**2-Anonymous Result:**
```
Age   | Gender | ZIP     | Disease
20-25 | M      | 0213*   | Asthma
20-25 | M      | 0213*   | Diabetes
25-30 | F      | 0214*   | Asthma
25-30 | F      | 0214*   | Heart Disease
```

**Verification:**
- (20-25, M, 0213*): 2 records ✓
- (25-30, F, 0214*): 2 records ✓

### Example 2: Information Loss Analysis

**Measuring Privacy vs Utility Trade-off:**

**Original Dataset Specificity:**
- Age: Exact values (high utility)
- ZIP: 5-digit codes (high utility)

**After 3-Anonymity:**
- Age: 10-year ranges (medium utility)
- ZIP: 3-digit prefixes (low utility)

**Utility Impact:**
- Statistical queries become less precise
- Age-based analysis loses granularity
- Geographic analysis loses local patterns

### Example 3: Attack Scenario

**Adversary Knowledge:**
- Target: John, 28-year-old male from ZIP 02141
- Public information: John visited hospital in March 2024

**Attack Process:**
1. Find records matching John's demographics
2. Use temporal information to narrow candidates
3. Infer sensitive medical condition

**Defense:** Ensure equivalence classes contain multiple individuals matching John's profile.

---

## ❓ Self-Assessment Questions

<details>
<summary><b>Question 1:</b> What is the difference between direct identifiers and quasi-identifiers? Provide examples. (Click to reveal answer)</summary>

**Answer:** 
**Direct identifiers** uniquely identify individuals on their own:
- Names, Social Security Numbers, email addresses, phone numbers, employee IDs

**Quasi-identifiers** don't uniquely identify individuals alone but can be combined for re-identification:
- Age, gender, ZIP code, date of birth, education level, occupation

The key difference is that removing direct identifiers is not sufficient for privacy protection because quasi-identifiers can be linked with external datasets to re-identify individuals.

</details>

<details>
<summary><b>Question 2:</b> A dataset has 100 records. After applying 5-anonymity, what is the maximum number of equivalence classes possible? (Click to reveal answer)</summary>

**Answer:** The maximum number of equivalence classes is 20.

**Reasoning:** If we have k-anonymity with k=5, each equivalence class must contain at least 5 records. With 100 total records, the maximum number of classes occurs when each class has exactly 5 records: 100 ÷ 5 = 20 classes.

In practice, the number might be less if some classes have more than 5 records to minimize information loss.

</details>

<details>
<summary><b>Question 3:</b> Consider this dataset. Does it satisfy 2-anonymity? If not, how would you fix it?

| Age | Gender | ZIP   | 
|-----|--------|-------|
| 23  | M      | 02139 |
| 24  | F      | 02139 |
| 25  | M      | 02140 |
| 26  | F      | 02140 |

(Click to reveal answer)</summary>

**Answer:** **No, it does not satisfy 2-anonymity.**

Each record has a unique combination of quasi-identifiers:
- (23, M, 02139) - appears 1 time
- (24, F, 02139) - appears 1 time  
- (25, M, 02140) - appears 1 time
- (26, F, 02140) - appears 1 time

**Solution - Generalize Age and ZIP:**
| Age   | Gender | ZIP   |
|-------|--------|-------|
| 20-25 | M      | 0213* |
| 20-25 | F      | 0213* |
| 25-30 | M      | 0214* |
| 25-30 | F      | 0214* |

Now we have two equivalence classes, each with 2 records, satisfying 2-anonymity.

</details>

<details>
<summary><b>Question 4:</b> What are the main limitations of k-anonymity that will be addressed in future weeks? (Click to reveal answer)</summary>

**Answer:** Main limitations of k-anonymity:

1. **Homogeneity Attack:** If all records in an equivalence class have the same sensitive value, privacy is not protected
2. **Background Knowledge Attack:** Attackers with additional information can still infer sensitive attributes
3. **Skewness Attack:** Sensitive attributes may not be well-represented, leading to probabilistic inference
4. **Lack of Diversity:** No guarantee that sensitive attributes are diverse within equivalence classes

These limitations led to the development of:
- **l-diversity:** Ensures diversity of sensitive attributes
- **t-closeness:** Ensures sensitive attribute distribution matches the overall population
- **Differential privacy:** Provides mathematical privacy guarantees

</details>

<details>
<summary><b>Question 5:</b> Explain the trade-off between privacy and utility in k-anonymity. How does increasing k affect this trade-off? (Click to reveal answer)</summary>

**Answer:** **Privacy vs Utility Trade-off:**

**Higher k (more privacy):**
- Larger equivalence classes provide better privacy protection
- Requires more generalization/suppression
- Results in greater information loss
- Statistical queries become less accurate
- Harder to perform fine-grained analysis

**Lower k (more utility):**
- Smaller equivalence classes preserve more data specificity
- Less generalization needed
- Higher data utility for analysis
- Reduced privacy protection
- Higher re-identification risk

**The Sweet Spot:** The optimal k value balances adequate privacy protection with acceptable utility loss for the intended data use cases. This depends on:
- Sensitivity of the data
- Threat model
- Required analysis tasks
- Available auxiliary information

</details>

---

## 📚 Additional Resources

### Foundational Papers
- Sweeney, L. (2002). "k-anonymity: a model for protecting privacy"
- Samarati, P. (2001). "Protecting respondents' identities in microdata release"

### Advanced Reading
- Fung, B. C. M. et al. (2010). "Privacy-preserving data publishing: A survey of recent developments"
- Ghinita, G. (2007). "Privacy for location-based services"

### Tools
- **ARX Data Anonymization Tool:** Open-source anonymization software
- **μ-ARGUS:** Statistical disclosure control tool
- **SECRETA:** Privacy-preserving data sharing platform

---

<div align="center">
   <div align="center">
  <a href="week1.md">⬅️ <strong>Main</strong></a> |
  <a href="README.md"><strong>Main</strong></a> |
  <a href="week2.md"><strong>Week 2</strong> ➡️</a>
</div>
