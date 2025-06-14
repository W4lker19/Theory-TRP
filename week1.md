# Week 1: Course Overview & Security/Privacy Concepts

<div align="center">
  <a href="https://w4lker19.github.io/Theory-TRP">⬅️ <strong>Main</strong></a> |
  <a href="week2.html"><strong>Week 2</strong> ➡️</a>
</div>


---

## 📚 Learning Goals

By the end of this week, you should understand:
- The fundamental concepts of security and privacy
- The distinction between privacy and confidentiality
- Basic threat models and attack scenarios
- The importance of privacy-enhancing technologies in modern digital systems

---

## 📖 Theoretical Content

### Security vs Privacy: Foundational Concepts

**Security** encompasses the protection of information and systems from unauthorized access, modification, or destruction. It traditionally focuses on the **CIA triad**:
- **Confidentiality:** Information is accessible only to those authorized
- **Integrity:** Information remains accurate and unaltered
- **Availability:** Information and systems are accessible when needed

**Privacy**, on the other hand, is about controlling how personal information is collected, used, and shared. Privacy is broader than confidentiality and includes:
- **Data minimization:** Collecting only necessary information
- **Purpose limitation:** Using data only for stated purposes
- **User control:** Giving individuals control over their data

### The Privacy Landscape

Modern digital systems create unprecedented privacy challenges:

**Data Collection Mechanisms:**
- Web tracking (cookies, fingerprinting)
- Mobile app permissions
- IoT device sensors
- Location services
- Social media interactions

**Privacy Threats:**
- **Inference attacks:** Deriving sensitive info from non-sensitive data
- **Linkage attacks:** Connecting datasets to re-identify individuals
- **Membership attacks:** Determining if someone's data is in a dataset
- **Reconstruction attacks:** Rebuilding original data from aggregates

### Privacy-Enhancing Technologies (PETs) Overview

PETs are technical and procedural measures designed to protect privacy while enabling legitimate data use:

1. **Anonymization Technologies**
   - Data anonymization and pseudonymization
   - k-anonymity, l-diversity, t-closeness

2. **Cryptographic Privacy**
   - Homomorphic encryption
   - Secure multiparty computation
   - Zero-knowledge proofs

3. **Differential Privacy**
   - Mathematical privacy guarantees
   - Noise injection mechanisms

4. **Privacy-Preserving Communication**
   - Anonymous communication networks
   - Mix networks and onion routing

5. **Authentication & Authorization**
   - Anonymous credentials
   - Attribute-based access control

### Legal and Regulatory Context

**GDPR (General Data Protection Regulation):**
- Privacy by design and by default
- Data subject rights (access, rectification, erasure)
- Consent and legitimate interest

**Privacy Principles:**
- **Proportionality:** Privacy measures should match the sensitivity
- **Transparency:** Clear communication about data practices
- **Accountability:** Organizations must demonstrate compliance

---

## 🔍 Detailed Explanations

### Understanding Threat Models

A **threat model** defines:
- **Assets:** What we want to protect (personal data, location, communications)
- **Threats:** Who might attack (governments, corporations, criminals)
- **Capabilities:** What attackers can do (intercept traffic, access databases)
- **Goals:** What attackers want to achieve (surveillance, profit, disruption)

**Example Threat Model - Web Browsing:**
- *Asset:* Browsing history and personal preferences
- *Threat:* Advertising companies and data brokers
- *Capabilities:* Track across websites using cookies and fingerprinting
- *Goal:* Build detailed profiles for targeted advertising

### Privacy vs Anonymity vs Pseudonymity

**Privacy:** Control over personal information disclosure
- Example: Choosing what to share on social media

**Anonymity:** Inability to identify an individual
- Example: Anonymous survey responses

**Pseudonymity:** Use of persistent identifiers that don't reveal real identity
- Example: Using a consistent username across platforms

### The Privacy Paradox

Users express concerns about privacy but often act in ways that compromise it:
- Accepting terms without reading them
- Sharing personal information for convenience
- Using free services that monetize personal data

This highlights the need for privacy-by-design approaches that protect users without requiring constant privacy decisions.

---

## 💡 Practical Examples

### Example 1: Web Tracking Mechanisms

**Scenario:** Understanding how websites track users

**Tracking Methods:**
1. **HTTP Cookies:** Small files stored in browsers
   ```
   Set-Cookie: user_id=12345; Expires=Wed, 09 Jun 2025 10:18:14 GMT
   ```

2. **Browser Fingerprinting:** Unique device characteristics
   - Screen resolution, installed fonts, plugins
   - Canvas fingerprinting, WebGL rendering

3. **Cross-Site Tracking:** Following users across different websites
   - Third-party cookies from ad networks
   - Social media buttons and analytics

**Privacy Implications:**
- Detailed behavioral profiles
- Price discrimination
- Filter bubbles and echo chambers

### Example 2: Location Privacy Scenario

**Scenario:** Smartphone location sharing

**Data Collection:**
- GPS coordinates with timestamps
- Wi-Fi and Bluetooth beacons
- Cell tower connections
- App-specific location requests

**Privacy Risks:**
- **Home/Work Inference:** Regular patterns reveal personal locations
- **Social Connections:** Co-location data reveals relationships
- **Lifestyle Inference:** Visited places indicate interests, health, etc.

**Protection Mechanisms:**
- Location obfuscation (adding noise)
- Caching and delayed reporting
- Anonymous authentication to services

---

## ❓ Self-Assessment Questions

<details>
<summary><b>Question 1:</b> What is the main difference between security and privacy? (Click to reveal answer)</summary>

**Answer:** Security focuses on protecting information and systems from unauthorized access, modification, or destruction (CIA triad: Confidentiality, Integrity, Availability). Privacy is broader and concerns controlling how personal information is collected, used, and shared, including concepts like data minimization, purpose limitation, and user control. Privacy includes confidentiality but extends beyond it to encompass user rights and data governance.

</details>

<details>
<summary><b>Question 2:</b> Explain the concept of a threat model and provide an example. (Click to reveal answer)</summary>

**Answer:** A threat model systematically identifies what we want to protect (assets), who might attack (threats), what attackers can do (capabilities), and what they want to achieve (goals). 

Example - Email Communication:
- *Assets:* Email content, sender/receiver identities, communication patterns
- *Threats:* Government surveillance, hackers, email providers
- *Capabilities:* Intercept network traffic, access email servers, analyze metadata
- *Goals:* Surveillance, espionage, identity theft, censorship

</details>

<details>
<summary><b>Question 3:</b> What are three main categories of privacy threats mentioned in the lecture? (Click to reveal answer)</summary>

**Answer:** 
1. **Inference attacks:** Deriving sensitive information from non-sensitive data
2. **Linkage attacks:** Connecting different datasets to re-identify individuals
3. **Membership attacks:** Determining whether someone's data is included in a particular dataset
4. **Reconstruction attacks:** Rebuilding original data from aggregate statistics

(Note: The question asked for three, but four main categories were covered)

</details>

<details>
<summary><b>Question 4:</b> How does differential privacy differ from traditional anonymization approaches? (Click to reveal answer)</summary>

**Answer:** Traditional anonymization (like k-anonymity) tries to prevent re-identification by modifying data, but can be vulnerable to auxiliary information attacks. Differential privacy provides mathematical guarantees by adding carefully calibrated noise to query results or datasets. It ensures that the inclusion or exclusion of any individual's data doesn't significantly change the probability of any outcome, providing provable privacy protection regardless of what background knowledge an attacker might have.

</details>

<details>
<summary><b>Question 5:</b> Describe the "privacy paradox" and its implications for system design. (Click to reveal answer)</summary>

**Answer:** The privacy paradox refers to the disconnect between users' stated privacy concerns and their actual behavior. Users often express strong privacy preferences but then act in ways that compromise privacy (accepting terms without reading, sharing personal information for convenience, using free services that monetize data). This highlights the need for privacy-by-design approaches that protect users automatically without requiring constant privacy decisions, rather than relying on user choice alone.

</details>

---

## 📚 Additional Resources

### Essential Reading
- Solove, D. J. (2008). "Understanding Privacy" - Harvard University Press
- Nissenbaum, H. (2009). "Privacy in Context" - Stanford University Press

### Technical Papers
- Sweeney, L. (2002). "k-anonymity: A model for protecting privacy"
- Dwork, C. (2008). "Differential Privacy: A Survey of Results"

### Tools and Frameworks
- **Privacy Badger:** Browser extension for tracking protection
- **Tor Browser:** Anonymous web browsing
- **Signal:** Private messaging application

### Legal Resources
- GDPR Official Text: [eur-lex.europa.eu](https://eur-lex.europa.eu/eli/reg/2016/679/oj)
- CCPA (California Consumer Privacy Act) Overview

---

<div align="center">
  <a href="https://w4lker19.github.io/Theory-TRP">⬅️ <strong>Main</strong></a> |
  <a href="week2.html"><strong>Week 2</strong> ➡️</a>
</div>
