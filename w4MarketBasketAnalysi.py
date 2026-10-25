import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

#explore
data = pd.read_csv('Online Retail.csv')
print(data.head())
print(data.info())

#no of transactions
print(len(data))

print(data.columns)

# useful colum for association ming inno identify transaction stpock code iedentify product description describ eproducut quanity is no of units purchase
#different regions
print(data.Country.unique())

#cleaning the data
#remove extra space from description

# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()
# Dropping the rows without any invoice number ##missing value will make it diffuclt to know which prodcuts belong to which invoice and products cannot be reliably grouped  the frequency of product combinatoons may nto be identiifed correctly
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)
# Convert invoice numbers to string # no calcualtion will be performed to enable string search to happen

data['InvoiceNo'] = data['InvoiceNo'].astype('str')

#Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]


#step 4 splitting transactions
#which producrt customers buy together in each country
basket_France= (data[data['Country'] =="France"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))
print(basket_France.shape) # (392,1563) 392 = invoice rows. 1563 = different product descriptions used as column so the unique products there

#transactions in uk
basket_uk= (data[data['Country'] =="United Kingdom"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))
print(basket_France.shape)

# Transactions done in Portugal
basket_Por = (data[data['Country'] =="Portugal"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))
print(basket_Por.shape)

#3) How many unique products do you think Portugal sells?”
# Transactions done in Sweden
basket_Sweden = (data[data['Country'] =="Sweden"]
.groupby(['InvoiceNo', 'Description'])['Quantity']
.sum().unstack().reset_index().fillna(0)
.set_index('InvoiceNo'))
print(basket_Sweden.shape) #(36, 261) 36 invoices 261 unique products

#In Market Basket Analysis, one-hot encoding converts your basket data into 1s and 0s so the Apriori algorithm can identify which products are bought together
#identify product bought togther and help generate associate rules
#Step 5: Hot encoding the Data

# Defining the hot encoding function to make the data suitable
# for the concerned libraries

def hot_encode(x):
    if(x<= 0):
        return 0 # o thik o mean not purchased one mean purchased
    if(x>= 1):
        return 1
    #if bought make it 1 if not 0 cause comp understand o AND 1
    #EG INVOICE 500 HAS 2 APPLE 0 PEAR SO BECOMES 1 0


# Encoding the datasets for each country basket
#old python version
#basket_encoded = basket_France.applymap(hot_encode) #applies function
#new python version
basket_encoded = basket_France.map(hot_encode)
basket_France = basket_encoded

#portugal
basket_encoded = basket_Por.map(hot_encode)
basket_Por= basket_encoded

#uk apply the encode to the full basket
basket_encoded = basket_uk.map(hot_encode)
basket_uk= basket_encoded

#sweeden
basket_encoded = basket_Sweden.map(hot_encode)
basket_Sweden= basket_encoded

#step 6 building the apricot

# Building the model for France
frq_items = apriori(basket_France, min_support = 0.05, use_colnames = True) # min support keepproduct combination that appear in atleast 5% of transactions
# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
# Sort and display strongest rules
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head(10))
print(rules.head(10).to_string())

#support 	How often are these products bought together?
#confidence :If someone buys the first product(s), how often do they also buy the other product?
#lift:How much more likely are they to be bought together than we'd expect by chance
# lift greater than 1 positive assoxiation =1 no association beyond whats expected by c chance less than 1 negTUIVE ASSOCIATION

#EXAMPLE INTERPRETATION This is a strong association. Customers who buy these cups and napkins are very likely to buy the matching plates too.
#Lift 1.26: The association is only about 1.26 times the expected rate under independence#
#Confidence 96.43%: 96.43% of transactions containing the picnic bag also contain postag
#
#INTERPRETATION: Confidence is high, but lift is relatively low. Postage appears in many transactions generally, so the high confidence does not necessarily mean the picnic bag has a particularly strong relationship with postage
#he Jumbo Bag Woodland Animals has a high confidence with postage, meaning every observed transaction containing this bag also included postage.
#However, the lift of 1.31 indicates only a relatively modest association, possibly because postage is already common in transactions.

#United Kingdom:
frq_items = apriori(basket_uk, min_support = 0.05, use_colnames = True) #SUPPORT MUST BE ATLEAST 10%
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

#INTERPRETATION
#f the rules for British transactions are analyzed a little deeper, it is seen that the
#British people buy different colored tea-plates together. A reason behind this may
#be because typically the British enjoy tea very much and often collect different
#colored tea-plates for different occasion

#Portugal:
frq_items = apriori(basket_Por, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())

#Sweden:
frq_items = apriori(basket_Sweden, min_support = 0.05, use_colnames = True)
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])
print(rules.head())