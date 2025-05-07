# Emotion Detection from Reddit Posts
https://arvindhraja-oracle-cloud-infrastructure-idzabfzxbdykivmvckf9fz.streamlit.app/


This project performs **emotion detection** on Reddit posts using the **RoBERTa** model from Hugging Face. It analyzes posts from the subreddit **r/depression**, extracts the emotional tone of the content, and visualizes the distribution of emotions using a pie chart.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies](#technologies)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project uses **Hugging Face's RoBERTa model** trained for emotion classification, combined with the **PRAW (Python Reddit API Wrapper)** to fetch posts from Reddit. The model classifies the emotion of each post into categories such as:
- **Anger**
- **Joy**
- **Optimism**
- **Sadness**
- **Fear**
- **Surprise**
- **Disgust**
- **Trust**

The emotional distribution is then visualized using a pie chart. The results are saved in a CSV file for further analysis.

## Features
- Fetches posts from the **r/depression** subreddit.
- Classifies text posts into 8 emotions: **anger**, **joy**, **optimism**, **sadness**, **fear**, **surprise**, **disgust**, and **trust**.
- Displays the results in the console.
- Saves the results in a CSV file.
- Creates a pie chart of the emotion distribution and saves it as an image.

## Technologies
- **Python**: Programming language used.
- **Hugging Face**: For pre-trained models and tokenization.
- **PRAW (Python Reddit API Wrapper)**: For accessing Reddit API and fetching posts.
- **PyTorch**: For running the model.
- **Matplotlib**: For visualizing the distribution of emotions as a pie chart.
- **pandas**: For data manipulation and saving results to CSV.

## Installation

Follow the steps below to set up this project on your local machine:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/emotion-detection-reddit.git
   cd emotion-detection-reddit
````

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The `requirements.txt` file should contain the following:

   ```plaintext
   praw
   torch
   transformers
   pandas
   matplotlib
   ```

4. Set up your **Reddit API credentials**. You can obtain these by registering a new application at [Reddit's App Preferences](https://www.reddit.com/prefs/apps):

   * **client\_id**: Your app's client ID.
   * **client\_secret**: Your app's secret.
   * **user\_agent**: A unique user agent.
   * **username** and **password**: Your Reddit username and password.

   Update the following lines in the code with your credentials:

   ```python
   reddit = praw.Reddit(
       client_id="your_client_id",
       client_secret="your_client_secret",
       user_agent="your_user_agent",
       username="your_reddit_username",
       password="your_reddit_password"
   )
   ```

## Usage

1. Run the script:

   ```bash
   python emotion_detection.py
   ```

2. The script will:

   * Fetch the latest 5 hot posts from the **r/depression** subreddit.
   * Analyze each post's emotion using the RoBERTa model.
   * Print the results (text and emotion) in the console.
   * Save the results to `reddit_emotions.csv`.
   * Generate a pie chart and save it as `emotion_pie_chart.png`.

3. Results will be saved in `reddit_emotions.csv` and the pie chart will be saved as `emotion_pie_chart.png` in the current directory.

## Results

Here’s an example of what the output will look like in the terminal:

```bash
🔍 Emotion Detection Results:

📝 Text: I feel really down today.
❤️ Emotion: sadness

📝 Text: I am so angry at myself for not achieving my goals.
❤️ Emotion: anger

📝 Text: I'm just here to check in and see how everyone else is doing.
❤️ Emotion: optimism

📝 Text: I don't know how to get past this sadness.
❤️ Emotion: sadness

📝 Text: Life just feels like it's getting harder every day.
❤️ Emotion: sadness

✅ Results saved to reddit_emotions.csv
```

## Contributing

Feel free to fork this repository, submit issues, and create pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

Thank you for using this project! Happy analyzing! 😊

```

### Explanation of the README:
1. **Overview**: Describes what the project does—emotion detection using RoBERTa on Reddit posts.
2. **Features**: Outlines what the script does (fetches posts, classifies emotions, saves results, and visualizes them).
3. **Technologies**: Lists the technologies used in the project.
4. **Installation**: Provides the necessary steps to set up the project on your local machine.
5. **Usage**: Instructions on how to run the script and what to expect in terms of output.
6. **Results**: A sample output of the script.
7. **Contributing**: Encourages others to contribute to the project.
8. **License**: The project is open-source under the MIT License.

This should help users understand the purpose, setup, and functionality of your project!
```
