# Ethics Considerations

## Customer profiling
Churn prediction builds **profiles** from billing, contract, and service usage data. Using scores to target customers can feel intrusive if not disclosed. Transparency about data use and opt-out options align with ethical marketing practice.

## Fairness concerns
Features such as **gender**, **senior citizen status**, and **payment method** may correlate with protected attributes or socioeconomic status. Models could **disadvantage subgroups** even without explicit intent. Fairness audits (equalized odds, demographic parity across groups) and careful feature governance are recommended before deployment.

## Automated decision making
High-risk uses—automatic service termination, punitive pricing, or reduced support priority based on churn score—raise **accountability** issues. Human review and appeal mechanisms should accompany automated actions.

## Potential discrimination
If retention offers are offered only to predicted churners in certain segments, **existing loyal customers** in other segments may receive worse treatment (**reverse discrimination** in loyalty programs). Electronic-check users and month-to-month customers may be over-targeted, reinforcing structural inequities.

## Privacy considerations
- **Personal identifiers** (`customerID`) must be excluded from models and reports shared externally.
- Deployed systems should follow **data minimization**, encryption, and access controls.
- Combining telco data with third-party sources increases re-identification risk.
- Compliance with GDPR/KVKK and local telecom regulations is mandatory for real deployments.

## Recommendations
Use churn scores for **supportive retention** (better offers, service fixes), not punishment. Document model purpose, limitations, and monitoring plans; involve legal/compliance review for production systems.
