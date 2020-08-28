# Pitchfork-Music-Analysis


<!--ts-->
   * [Table of Contents](#table-of-contents)
   * [Introduction](#introduction)
   * [Conclusions](#conclusions)
   * [Future Steps](#future-steps)
<!--te-->
## Introduction
Pitchfork is a music and arts website that was created in 1995 by Ryan Schieber. From its humble beginnings based off of a blog Ryan made as a music store clerk it has since been bought by Conde Nast and is now a preeminant voice in the music scene. In this analysis I plan to use this [Kaggle](https://www.kaggle.com/nolanbconaway/pitchfork-data) dataset to analyize text data of reviews from the past ~18 years to see how the voice of the website has changed over time. 
### Data Introduction
The dataset I chose is a sqlite database with tables titled "Artists", "Content", "genres","labels","reviews",and "years". There were thankfully few null values, as the data was taken directly from Pitchforks Website using beautiful soup. In the dataset there were 18,000 reviews from 1999 to 2018
## Conclusions

First I was able to create a system to label topics and find word frequency respective to the reviews authors. From that I was able to create a model to describe text based reviews with some amount of accuracy. 

## Future Steps

In the future I would like to improve the predicitablity of the model to determine genre type. In addition I would like to compare reviews from other popular music review sites, such as Rolling Stone, Billboard, or even finding a way to transcribe video reviews and analyze TheNeedleDrop's youtube channel.