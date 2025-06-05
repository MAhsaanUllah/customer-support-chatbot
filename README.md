Customer Support Automation Chatbot 🤖

A web-based chatbot that automates customer support responses using NLP and machine learning. It processes queries, identifies sentiments (positive, negative, neutral), and generates empathetic replies using GPT-2. Deployed on Streamlit Cloud for easy access.

🚀 Features





Sentiment Analysis: Categorizes customer queries into positive, negative, or neutral using VADER and TextBlob.



Empathetic Responses: Generates tailored replies with GPT-2, with fallback responses for irrelevant outputs.



Issue Trends Visualization: Displays customer issue distribution using Seaborn bar charts.



Live Demo: Hosted on Streamlit Cloud.

🛠️ Tech Stack





Languages: Python



Frameworks: Streamlit (web interface)



Libraries: Transformers (GPT-2), Scikit-learn (logistic regression, TF-IDF), NLTK, TextBlob, VADER (sentiment analysis), Matplotlib, Seaborn (visualization), PyTorch



Deployment: Streamlit Cloud

📊 How It Works





Data Loading: Loads a public dataset for training (https://raw.githubusercontent.com/zfz/twitter_corpus/master/full-corpus.csv).



Text Preprocessing: Cleans text by removing URLs, non-alphabetic characters, and converting to lowercase.



Model Training: Uses TF-IDF vectorization and logistic regression to classify issues.



Sentiment Detection: Analyzes sentiment with VADER and enhances with custom negative indicators (e.g., "haven’t received").



Response Generation: GPT-2 generates empathetic replies with a 10-second timeout; falls back to defaults if needed.



Visualization: Plots issue trends for insights.

🧪 Testing Results





"My order is delayed and I am very disappointed!" → Negative: "We’re sorry for the delay..."



"I love your service, it’s so fast and amazing!" → Positive: "Thank you for your kind words..."



"My package arrived but it’s damaged." → Negative: "We’re sorry for the damaged item..."



"Can you tell me about your refund policy?" → Neutral: "For refund policy, contact support@example.com."



"I haven’t received my order yet." → Negative: "We’re sorry for the missing order..."

📦 Installation





Clone the repository:

git clone https://github.com/your-username/customer-support-chatbot.git



Install dependencies:

pip install -r requirements.txt



Run the app locally:

streamlit run app.py

🌐 Deployment





Deployed on Streamlit Cloud: Live Demo



To redeploy:





Commit changes to GitHub.



Go to Streamlit Cloud, select the app, and click "Redeploy".



Wait 5-10 minutes.

🔮 Future Improvements





Add multi-language support.



Upgrade to advanced models like GPT-3.



Collect user feedback for better accuracy.



Create interactive dashboards.

📝 About the Developer

Hi! I’m Muhammad Ahsaan Ullah, a university student passionate about AI, machine learning, and web development. This project showcases my skills in building real-world applications. Connect with me on LinkedIn or check out my other projects on GitHub.



Built with ❤️ and ☕ in Pakistan
