Workflow:
1. Data analysis - data_analysis_healthcare_no_show.ipynb
2. Data preparation for training - dataset_prep_healthcare_no_show.ipynb
3. Modeling - modeling_healthcare_no_show.ipynb

Data representations for machine learning:
1. patient_id
    - no need
2. appointment_id
    - no need
3. gender
    - encode as binary with 0 as male and 1 as female
4. scheduled_day
    - calculate lag days with appointment_day, remove negative, and use min-max normalization
    - may group them into ["same day", "7 days", "14 days", "30 days", "60 days", "90 days", ">90 days"] and use one-hot encoding
5. appointment_day
    - convert to day of week and use one-hot encoding
    - also see scheduled_day
6. age
    - remove negatives
    - use min-max normalization
    - may group into ["Infant", "Toddler", "Child", "Teen", "Adult", "Middle", "Senior"] and use one-hot encoding
7. neighbourhood
    - use one-hot encoding
    - may not be useful and can ignore
8. scholarship
    - encode as binary
9. hypertension
    - encode as binary
10. diabetes
    - encode as binary
11. alcoholism
    - encode as binary
12. handicap
    - use one-hot encoding
    - may normalize to binary with 0 as no handicap and 1 as have handicap
13. sms_received
    - encode as binary
14. no_show
    - output
    - encode as binary