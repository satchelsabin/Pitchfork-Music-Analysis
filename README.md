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
The dataset I chose is a sqlite database with tables titled "Artists", "Content", "genres","labels","reviews",and "years". There were thankfully few null values, as the data was taken directly from Pitchforks Website using beautiful soup. In the dataset there were 18,000 reviews from 1999 to 2018.
### Introductory Analysis

Longest:['andy beta', 'mark richardson', 'matt lemay', 'rob mitchum', 'joe tangari', 'amanda petrusich', 'julianne escobedo shepherd', 'brandon stosuy', 'philip sherburne', 'brian howe']





Prolific:['joe tangari', 'stephen m. deusner', 'ian cohen', 'brian howe', 'mark richardson', 'stuart berman', 'marc hogan', 'nate patrin', 'marc masters', 'jayson greene']



### Text Analysis


LongestDf: Overall Impactful
['burma', 'dylan', 'drake', 'guitar', 'benson', 'adams', 'rock',
       'jones', 'cohen', 'beam']
ProlificDf: Overall Impactful
['mccartney' 'springsteen' 'cheese' 'christmas' 'elvis' 'white' 'beep'
 'adams' 'oasis' 'fahey']
 
 LongestDf Word Cloud
 
 ProlificDF Word Cloud


### Feature Extraction

Topic List:
reconstruction error: 42.66767111761127
topic 0
--> rap like hip hop he's there's even get beats one i'm still way enough good that's time got record beat
topic: rap

topic 1
--> disc tracks soul music set funk live compilation two version original years one material early label jazz new recorded first
topic: soul

topic 2
--> music like sound sounds electronic guitar tracks piano noise ambient record work piece track drone pieces one something way bass
topic: electronic

topic 3
--> band rock band's album guitar like punk song sound bands guitars even one songs time post much though they're they've
topic: rock

topic 4
--> songs album song like sings voice sounds country folk lyrics even sound guitar one life he's love singing acoustic new
topic: folk

topic 5
--> pop indie love like rock new dance punk debut synth house girl self disco post us beach electro still also
topic: pop

Examples:
eula wool sucking
--> 88.77% Rock
--> 0.46% Electronic
--> 0.22% Rap
--> 10.35% Soul/Funk
--> 0.05% Country/Songwriter
--> 0.06% Pop/Dance
--> 0.09% Guitar

king krule 6 feet beneath the moon
--> 0.07% Rock
--> 5.15% Electronic
--> 0.07% Rap
--> 9.83% Soul/Funk
--> 0.07% Country/Songwriter
--> 0.07% Pop/Dance
--> 84.73% Guitar

crocodiles crimes of passion
--> 0.00% Rock
--> 0.16% Electronic
--> 99.80% Rap
--> 0.01% Soul/Funk
--> 0.00% Country/Songwriter
--> 0.01% Pop/Dance
--> 0.00% Guitar



## Conclusions

First I was able to create a system to label topics and find word frequency respective to the reviews authors. From that I was able to create a model to describe text based reviews with some amount of accuracy. 

## Future Steps

In the future I would like to improve the predicitablity of the model to determine genre type. In addition I would like to compare reviews from other popular music review sites, such as Rolling Stone, Billboard, or even finding a way to transcribe video reviews and analyze TheNeedleDrop's youtube channel.