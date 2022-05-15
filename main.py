from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
from scipy.__config__ import show
from PIL import Image
import plotly.express as px
import streamlit as st
from DecisionTree import decision_trees, fig_visualization_binary, fig_visualization_dt
from LocalSensitivityHashing import print_results
from LogisticRegression import fig_visualization_lr, logistic_regression
from NaiveBayes import fig_visualization,cnb, fig_visualization_nb_binary
from utils import *
import plotly.graph_objs as go
from collections import Counter


st.title("Twitter Sentiment Analysis")
st.subheader("A data mining approach.")


df = clean_dataset()
data_pos=df[df['sentiment']=='1']['cleaned_tweets']
data_neg=df[df['sentiment']=='0']['cleaned_tweets']
data_neu=df[df['sentiment']=='2']['cleaned_tweets']
df['temp_list'] = df['cleaned_tweets'].apply(lambda x:str(x).split())
pos_mask = np.array(Image.open('twitterimage.jpg'))
data_positive = df[df['sentiment']=='1']
data_negative = df[df['sentiment']=='0']
data_neutral = df[df['sentiment']=='2']
decode_map = {0: "Negative", 2: "Neutral", 1: "Positive"}

def show_dataframe(df):

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Tweet</b>',
                    '<b>Cleaned Tweets</b>',
                    '<b>Sentiment</b>'],
            line_color='rgb(29 ,161, 242)', fill_color='rgb(225, 232, 237)',
            align='center', font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[df['text'], df['cleaned_tweets'], df['sentiment']],
            line_color='rgb(101 ,119 ,134)',
            fill_color='rgb(29 ,161, 242)',
            align='center', font=dict(color='white', size=10)
        ))
    ])
    fig.add_annotation(dict(font=dict(color='gray', size=17),
                            x=0,
                            y=-0.12,
                            showarrow=False,
                            text="This is what we will be using. Negative is 0, Positive is 1, Neutral is 2.",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    return fig

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), color = 'white',
                   title = None, title_size=40, image_color=False):
    stopwords = set(STOPWORDS)
    more_stopwords = {'u', "im"}
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color=color,
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    width=400, 
                    height=200,
                    mask = mask)
    wordcloud.generate(str(text))
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'black', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()  

def most_common_words_all(df):
    top = Counter([item for sublist in df['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words','count']
    temp.style.background_gradient(cmap='Blues')

    fig = px.treemap(temp, path=['Common_words'], values='count',title='Most Common words through our dataset')
    return fig

def most_common_words(data):
    top = Counter([item for sublist in data['temp_list'] for item in sublist])
    temp = pd.DataFrame(top.most_common(20))
    temp.columns = ['Common_words','count']
    temp.style.background_gradient(cmap='Greens')
    fig4 = px.bar(temp, x="count", y="Common_words", title='Most Commmon Words', orientation='h', 
                width=700, height=700,color='Common_words')
    return fig4

def decode_sentiment(label):
    return decode_map[int(label)]

def plot_statistics(df):
    df.sentiment = df.sentiment.apply(lambda x: decode_sentiment(x))
    target_cnt = Counter(df.sentiment)
    x_axis=df.sentiment
    y_axis=list(target_cnt.values())
    fig = go.Figure([go.Bar(x=x_axis, y=y_axis)])
    return fig
   


st.subheader("How does our data look like?")
st.plotly_chart(show_dataframe(df))
st.subheader("What are we working with?")
st.plotly_chart(plot_statistics(df))


fig1 =plot_wordcloud(data_pos,mask=pos_mask,color='white',max_font_size=100,title_size=30,title="Positive tweets")
st.pyplot(fig1)
st.set_option('deprecation.showPyplotGlobalUse', False)

fig2 =plot_wordcloud(data_neg,mask=pos_mask,color='white',max_font_size=100,title_size=30,title="Negative tweets")
st.pyplot(fig2)
st.set_option('deprecation.showPyplotGlobalUse', False)

fig3 =plot_wordcloud(data_neu,mask=pos_mask,color='white',max_font_size=100,title_size=30,title="Neutral tweets")
st.pyplot(fig3)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader("Distribution of the most common words through the dataset")
st.plotly_chart(most_common_words_all(df))


st.subheader("Our distribution of positive words:")
st.plotly_chart(most_common_words(data_positive))
st.subheader("Our distribution of negative words:")
st.plotly_chart(most_common_words(data_negative))
st.subheader("Our distribution of neutral words:")
st.plotly_chart(most_common_words(data_neutral))

st.header('Lets check models')
st.subheader('Naive Bayes with Neutral/Positive/Negative (number of records)')
fig_naive = fig_visualization()
st.pyplot(fig_naive)
st.subheader("Precision Score")
st.subheader("Train accuracy:")
st.write(cnb()[1])
st.subheader("Test accuracy:")
st.write(cnb()[2])


st.subheader('Naive Bayes with Positive/Negative (percentage)')
fig_naive_bi = fig_visualization_nb_binary()[0]
st.pyplot(fig_naive_bi)
st.subheader("Precision Score")
st.subheader("Train accuracy:")
st.write(fig_visualization_nb_binary()[1])
st.subheader("Test accuracy:")
st.write(fig_visualization_nb_binary()[2])


st.subheader('Decision Trees with Positive/Negative values (percentage)')
fig_dt = fig_visualization_binary()[0]
st.pyplot(fig_dt)
st.subheader("Test accuracy:")
st.write(fig_visualization_binary()[1])


st.subheader('Decision Trees with Positive/Negative/Neutral values (number of records)')
fig_dt1 = fig_visualization_dt()
st.pyplot(fig_dt1)
st.subheader("Precision Score")
st.subheader("Test accuracy:")
st.write(decision_trees()[1])


st.subheader('Logistic Regression (percentage)')
fig_lr = fig_visualization_lr()
st.pyplot(fig_lr)
st.subheader("Precision Score")
st.subheader("Train accuracy:")
st.write(logistic_regression()[1])
st.subheader("Test accuracy:")
st.write(logistic_regression()[2])

st.subheader("Local Sensitivity Hashing")
tweet = st.text_input("Tweet I wanna see sentiment")
st.subheader("This tweet is positive with:")
st.subheader(print_results(tweet)[0].values)
st.subheader("This tweet is negative with:")
st.subheader(print_results(tweet)[1].values)
st.subheader("This tweet is neutral with:")
st.subheader(print_results(tweet)[2].values)