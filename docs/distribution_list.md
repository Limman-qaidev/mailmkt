# Distribution Lists and Recommendation Model

This guide explains how the dashboard builds distribution lists from past campaign data.

## Model Overview

The recommendation engine lives in `email_marketing.analytics` and uses a logistic regression
model trained on historical campaign engagement.

1. **Feature generation** – For each campaign ID, the `features` module
   assembles per-recipient statistics such as open counts, click counts and
   time to first open.
2. **Model scoring** – The trained model predicts the probability that a
   recipient will engage with the campaign.
3. **Thresholding** – Only recipients with a probability above the selected
   *recommendation threshold* are included in the resulting list.

## Generating Lists in the Editor

On the *Email Campaign Editor* page choose **By campaign type** as the recipient
source. Provide the desired *Campaign ID* and adjust the *Recommendation
threshold*. Use the **Preview** button to calculate the list.

The preview allows filtering by email domain to simulate client segmentation.
After previewing and filtering, the list is used when sending the campaign.

## Re-training

From the sidebar you can retrain the model or recalibrate feature weights. This
updates the recommendations for subsequent campaigns.