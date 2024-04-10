# cs-451-quant-project


## Abstract
Few investors are able to consistently beat stock indices like the S&P 500 regardless of whether they use quantitative approaches or not. We would like to contribute the latest attempt at predicting stock prices accurately nonetheless. We plan to build a model able to predict stock price movements using a mix of fundamental and “alternative” data, which is any relevant data that impacts a company’s financials but are not included in traditional sources of company information. We will assess our performance in multiple ways, such as simple accuracy measures (prediction vs. reality) and potentially with trading simulations in which our algorithm “invests” over a period of time and we compare its performance to benchmarks like the S&P 500.


## Motivation and Question
We believe that basic social media data is an underappreciated asset in trying to predict share price movements. Many investors today continue to look purely at financial metrics, but given how much information is accessible on the internet now largely for free, we think adding alternative data on top of that may give our model a high degree of accuracy.


## Planned Deliverables
We expect our project to have the standard deliverables:
- A Python package containing all code used for algorithms and analysis, including documentation.
- At least one Jupyter notebook illustrating the use of the package to analyze data.


Evaluating Effectiveness:
- Full Success: We will have a model that uses financial and alternative data to make predictions with consistently better-than-random accuracy on multiple companies. It will have accuracy better than random chance on companies outside of training data.
- Partial Success: We have a model useful for only a few companies. It isn't very useful outside of those companies, but it still has better-than-random accuracy.


## Resources Required
We intend to use historical US stock price data from Yahoo and Refinitiv, both of which are readily available to us (latter is available through Microsoft Excel). SocialBlade is a free platform for tracking social media. Its data is somewhat limited in timeframe (usually monthly or weekly). Google Trends is also readily accessible and will be a key source of information for us.


Company financial information will be more difficult to access in bulk. Company filings aren’t neatly organized for data analysis / machine learning purposes, so we’ll have to turn to other sources. It will be easy to find a single company’s financial data, but it will be relatively slow, labor-intensive work. We aren’t yet sure how we’ll efficiently access financial data in bulk.


## What You Will Learn


James
Goal: “Create a high quality, easy-to-understand project I can refer back to in the future for personal projects down the line”

I’m interested in keeping a clean code base that is well-commented and well-organized throughout the project so it can be a useful reference for future work. I’d like to use this project as an opportunity to apply what I know conceptually about finance in a model and test it rigorously. I’m interested in also getting familiar with cleaning up messy real-world data so that I get experience beyond the clean datasets we’re generally provided in classroom settings.


Donovan
Through working on this project, I wish to learn more about the financial markets, quantitative aspects that factor into trade, increase my coding repertoire (specifically in relation to quantitative trading), create meaningful relationships with my project partners, ensure I have a full grasp of the model we create (knowing what we are doing and why completely) and hopefully create a successful model by our groups metrics. In order to accomplish these feats, not only will I have to communicate effectively and proactively with my group mates, but also stay interested and motivated throughout. Tying this back to my original goal-setting reflection that I completed near the beginning of the semester, this aspect of the project does not faze me due to my own personal interest in the matter. From the very beginning of signing up for this class, my interest has always been to develop a model such as the one we are proposing. I hope that this aspiration continues to grow and develop as the project comes to fruition. 


## Risk Statement
The greatest risk is likely that we'll end up with insignificant results. The financial markets are extremely competitive, and professional traders and researchers spend hours every day trying to do exactly the same thing as us but with greater knowledge, time, and resources. 


To believe that the creation of a consistently accurate model is even possible to create in the first place is to believe that financial markets are not perfectly efficient, which goes against common wisdom among academic financial experts.


## Ethics Statement
In every trade, there is a winner and a loser. So in public markets, we are inherently creating a group that benefits and a group that does not benefit. If successful, our project may advantage the users of this algorithm in public markets and exclude from benefit anyone who takes opposing trades.


We aren’t entirely sure the world will be a better place with our project, but financial markets may be just slightly more accurate. The function of public markets is price discovery: what are companies worth? If our algorithm is able to answer that question slightly more accurately than individuals and other algorithms, we will make a model that is both profitable and useful to the world.


## Tentative Timeline
3-Week Check-in: Have a working regression model that can take in 1-2 types of social media data and output a predicted share price for a particular company or a small set of companies.


6-Week Check-in: Show the performance of our model across a wide range of companies in a wide range of market environments (e.g., how does our model do during the 2020 COVID stock market crash? Probably not very well, but it’s worth knowing how it reacts to black swan events). Build out rigorous testing and analysis to determine whether our model is accurate enough to be useful.
