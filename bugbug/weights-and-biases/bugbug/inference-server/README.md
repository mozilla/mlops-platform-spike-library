## Setting up the server:

1. Navigate to the inference server location: `$cd mlops-platform-spike-library/bugbug/weights-and-biases/bugbug/inference-server`
2. Install dependencies: `$pipenv install`
3. Launch local environment: `$pipenv shell`
4. Start the server: `$serve run wandb_artifact_inference_example:main_deployment`

**You'll know it succeeded when**:

You see your console output display the text: "Deployed Serve app successfully."

### At this point you can:
- See your node dashboard at `127.0.0.1:8265`
- Check out an interactive call UI at `127.0.0.1:8000`

## Using the server:
1. Visit the interactive call UI at `127.0.0.1:8000` in your browser.
2. Click the dropdown next to `POST /spambug_prediction`.
3. Click the 'Try it Out' button.
4. Enter a request body containing a list of bugs. You can use this request body with these bug IDs if you like:
```json
{
  "bug_ids": [
    1567822, 1604642
  ]
}
```
or you can get your own bug IDs [right from bugzilla](https://bugzilla.mozilla.org/buglist.cgi?product=Firefox&component=about%3Alogins&resolution=---).

5. Click the 'Execute' button.

**You should be able to scroll down and see a successful response like:**

```json
{
  "bug_probabilities": [
    {
      "bug_id": 1567822,
      "summary": "about:logins should re-use an already open tab",
      "creator_detail": {
        "name": "hskupin@gmail.com",
        "real_name": "Henrik Skupin [:whimboo][⌚️UTC+2]",
        "email": "hskupin@gmail.com",
        "id": 76551,
        "nick": "whimboo"
      },
      "probability_legitimate_bug": 0.9996013641357422,
      "probability_spam_bug": 0.0003986297524534166
    },
    {
      "bug_id": 1604642,
      "summary": "After searching for a login, Firefox Lockwise does not sort A-Z properly",
      "creator_detail": {
        "real_name": "distraida1973@gmail.com",
        "name": "distraida1973@gmail.com",
        "id": 653310,
        "nick": "distraida1973",
        "email": "distraida1973@gmail.com"
      },
      "probability_legitimate_bug": 0.9525588750839233,
      "probability_spam_bug": 0.04744110628962517
    }
  ]
}
```




