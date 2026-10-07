# Day 3 — Lexical Retrieval Baseline

## Goal

Evaluate the current lexical retrieval system using the 15 development questions created during the evaluation setup.

## Baseline

The system retrieves policy passages using lexical overlap and presents them for human review. It does not generate AI answers.

## Results

| Category | Questions | Observation |
|---|---:|---|
| Direct | Q01–Q06 | Generally retrieved the expected evidence |
| Paraphrased | Q07–Q09 | Mixed performance |
| Unanswerable | Q10–Q12 | No unsupported AI answers; some irrelevant evidence retrieved |
| Ambiguous | Q13 | Did not ask for clarification |
| Conflicting | Q14–Q15 | Mixed conflict retrieval |

## Important failures

- Q08: Retrieved unrelated access-control evidence instead of transit-encryption evidence.
- Q09: Found the backup policy but retrieved backup frequency instead of restore-testing evidence.
- Q13: Retrieved retention evidence instead of asking the user to clarify the ambiguous question.
- Q14: Retrieved the 90-day retention evidence but initially missed the conflicting 30-day evidence.

## What worked

- Direct questions were generally handled well.
- The system retrieved exact supporting passages for many questions.
- The system does not generate unsupported AI answers.
- Q15 successfully surfaced both conflicting retention statements.

## Conclusion

The lexical baseline provides a useful starting point but struggles when the question is paraphrased, ambiguous, or requires connecting multiple pieces of evidence.

## Next

Add an embedding-based retriever while keeping the lexical baseline unchanged so both approaches can be compared using the same evaluation questions.