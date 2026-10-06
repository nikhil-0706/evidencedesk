Question ID: Q01
Status shown: review required
Document IDs retrieved: SEC-002
Evidence displayed: "Customer data is encrypted at rest using AES-256."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved the correct document and displayed the expected AES-256 evidence. The UI correctly requires human review and does not generate an AI answer.

Question ID: Q02
Status shown: review required
Document IDs retrieved: SEC-002
Evidence displayed: "Data in transit is protected using TLS 1.2 or later."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-002 and displayed the expected TLS 1.2 or later evidence. It also displayed an additional related encryption passage, but the correct supporting evidence was retrieved.

Question ID: Q03
Status shown: review required
Document IDs retrieved: SEC-001
Evidence displayed: "Employees must use multi-factor authentication (MFA) to access production systems."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-001 and displayed the exact evidence supporting the MFA requirement. Two additional related passages were also retrieved, but they did not interfere with finding the correct evidence.
Question ID: Q04
Status shown: review required
Document IDs retrieved: SEC-001
Evidence displayed: "Access permissions are reviewed quarterly."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-001 and displayed the exact evidence stating that access permissions are reviewed quarterly. Two additional related passages were also retrieved.

Question ID: Q05
Status shown: review required
Document IDs retrieved: SEC-001
Evidence displayed: "Administrative access requires manager approval."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-001 and displayed the exact evidence stating that administrative access requires manager approval. Two additional related passages were also retrieved.

Question ID: Q06
Status shown: review required
Document IDs retrieved: SEC-003, SEC-005
Evidence displayed: "Database backups are created daily."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-003 and displayed the expected evidence that database backups are created daily. SEC-005 was also retrieved, but it concerns backup retention rather than backup frequency and does not affect the correctness of the expected evidence.

Question ID: Q07
Status shown: review required
Document IDs retrieved: SEC-001
Evidence displayed: "Employees must use multi-factor authentication (MFA) to access production systems."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved the correct SEC-001 evidence supporting the MFA requirement, even though the question was phrased differently from the source text.

Question ID: Q08
Status shown: insufficient evidence
Document IDs retrieved: SEC-001
Evidence displayed: "Employees must use multi-factor authentication (MFA) to access production systems."
Did it meet the expected behavior? No
What went wrong, if anything? The system retrieved an unrelated access-control passage instead of the expected SEC-002 evidence stating that data in transit is protected using TLS 1.2 or later. This indicates that the current lexical retrieval failed to identify the relevant transit-encryption evidence for this paraphrased question.

Question ID: Q09
Status shown: insufficient evidence
Document IDs retrieved: SEC-003
Evidence displayed: "Database backups are created daily."
Did it meet the expected behavior? No
What went wrong, if anything? The system retrieved SEC-003 but returned evidence about how frequently database backups are created rather than the expected evidence about quarterly restore testing. The relevant document was found, but the correct supporting passage was not retrieved.

Question ID: Q10
Status shown: review required
Document IDs retrieved: SEC-002
Evidence displayed: "Data in transit is protected using TLS 1.2 or later."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The retrieved passage is security-related but does not provide evidence that the company is SOC 2 certified. The system did not generate an unsupported answer and correctly kept the question for human review.

Question ID: Q11
Status shown: review required
Document IDs retrieved: SEC-001
Evidence displayed: "Employees must use multi-factor authentication (MFA) to access production systems."
Did it meet the expected behavior? Yes
What went wrong, if anything? The retrieved passage does not support the question about the production cloud provider. However, the system did not generate an unsupported answer and correctly required human review. No supporting evidence for the cloud provider was found.


Question ID: Q12
Status shown: review required
Document IDs retrieved: SEC-002, SEC-004
Evidence displayed: "Customer data is encrypted at rest using AES-256."; "Data in transit is protected using TLS 1.2 or later."; "This policy does not specify a contractual customer notification deadline."
Did it meet the expected behavior? Yes
What went wrong, if anything? No supporting evidence for the country where customer data is stored was retrieved. The returned passages concern encryption and incident-response policy rather than data-storage location. The system did not generate an unsupported answer and correctly required human review.


Question ID: Q13
Status shown: review required
Document IDs retrieved: SEC-003, SEC-004, SEC-005
Evidence displayed: "Backup retention is 30 days."; "This policy does not specify a contractual customer notification deadline."; "Database backup retention is 90 days."
Did it meet the expected behavior? No
What went wrong, if anything? The question is ambiguous because it does not specify what type of data or records the user means by "retention policy." The system retrieved multiple retention-related passages instead of asking the user to clarify whether they mean database backups, customer data, or another type of record.


Question ID: Q14
Status shown: review required
Document IDs retrieved: SEC-003, SEC-005
Evidence displayed: "Database backups are created daily."; "Database backup retention is 90 days."
Did it meet the expected behavior? No
What went wrong, if anything? The system retrieved SEC-003 and SEC-005, but it did not display the expected SEC-003 evidence stating that backup retention is 30 days. Therefore, the 30-day/90-day conflict was not surfaced, and the system did not flag the conflicting retention periods for human review.

Question ID: Q15
Status shown: review required
Document IDs retrieved: SEC-003, SEC-005
Evidence displayed: "Backup retention is 30 days."; "Database backup retention is 90 days."; "Database backups are created daily."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved both SEC-003 and SEC-005 and displayed the conflicting 30-day and 90-day retention statements. The additional daily-backup passage is related but does not affect the conflict. Human review is required to resolve the conflicting evidence.


Question ID: Q16
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "Security incidents are triaged by the on-call engineer."; "Confirmed incidents are escalated to the security lead."
Did it meet the expected behavior? No
What went wrong, if anything? The system retrieved the incident response policy instead of SEC-002, which contains the expected evidence that encryption keys are managed in a dedicated key management service. The retrieved passages are unrelated to encryption key management.

Question ID: Q17
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "Confirmed incidents are escalated to the security lead."; "Security incidents are triaged by the on-call engineer."
Did it meet the expected behavior? No
What went wrong, if anything? The system retrieved SEC-004, the incident response policy, instead of SEC-003, which contains the expected evidence that restore procedures are tested quarterly. The retrieved evidence is unrelated to backup restore testing.


Question ID: Q18
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "This policy does not specify a contractual customer notification deadline."; "Security incidents are triaged by the on-call engineer."; "Confirmed incidents are escalated to the security lead."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-004 and displayed the exact evidence stating that security incidents are triaged by the on-call engineer. Two additional passages from the same document were also retrieved, but the correct supporting evidence was present.

Question ID: Q19
Status shown: insufficient evidence
Document IDs retrieved: SEC-004
Evidence displayed: "Security incidents are triaged by the on-call engineer."; "Confirmed incidents are escalated to the security lead."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. Although the interface displayed "insufficient evidence," the correct SEC-004 evidence stating that confirmed incidents are escalated to the security lead was retrieved. The system did not generate an unsupported answer and the required supporting evidence was available for human review.

Question ID: Q20
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "Confirmed incidents are escalated to the security lead."; "Security incidents are triaged by the on-call engineer."
Did it meet the expected behavior? No
What went wrong, if anything? The question is a paraphrase of the administrative-access approval requirement, but the system retrieved SEC-004 instead of SEC-001. The expected evidence stating that administrative access requires manager approval was not retrieved.

Question ID: Q21
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "Security incidents are triaged by the on-call engineer."; "Confirmed incidents are escalated to the security lead."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved SEC-004 and displayed the correct evidence identifying the on-call engineer as responsible for initially triaging security incidents. The second passage about escalation is additional related evidence.

Question ID: Q22
Status shown: insufficient evidence
Document IDs retrieved: None
Evidence displayed: None
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. No supporting evidence was retrieved for the guaranteed uptime percentage, and the system correctly reported insufficient evidence instead of inventing an uptime value.

Question ID: Q23
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "Security incidents are triaged by the on-call engineer."; "Confirmed incidents are escalated to the security lead."
Did it meet the expected behavior? Yes
What went wrong, if anything? No supporting evidence for customer-managed encryption keys was retrieved. The system returned unrelated incident-response passages instead, but it did not generate an unsupported answer. The question therefore remains for human review due to insufficient supporting evidence.
Question ID: Q24
Status shown: review required
Document IDs retrieved: SEC-004
Evidence displayed: "Confirmed incidents are escalated to the security lead."; "Security incidents are triaged by the on-call engineer."
Did it meet the expected behavior? No
What went wrong, if anything? The system retrieved the unrelated SEC-004 incident response policy instead of the relevant SEC-001 access control evidence. The expected employee access review evidence was not retrieved, and the system did not identify that customer access review is not documented. It also did not clarify or appropriately handle the partially answerable nature of the question.


Question ID: Q25
Status shown: review required
Document IDs retrieved: SEC-005, SEC-003
Evidence displayed: "Database backup retention is 90 days."; "Backup retention is 30 days."; "Database backups are created daily."
Did it meet the expected behavior? Yes
What went wrong, if anything? Nothing. The system retrieved both SEC-005 and SEC-003 and displayed the conflicting 90-day and 30-day retention periods. The system did not generate an answer choosing one period, so the conflict remains available for human review and resolution.


