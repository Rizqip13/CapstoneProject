import pandas as pd

def infotable(x):
    return pd.DataFrame(columns='dataFeatures dataType null nullPct unique uniqueSample'.split(),
                        data=[[col, 
                               x.dtypes[col], 
                               sum(x[col].isna()),
                               round(100*sum(x[col].isna())/len(x[col]),2), 
                               x[col].nunique(),list(x[col].unique()[:5])]
                              for col in x.columns])       

def variabletypes(dataType, unique):
    if 'object' == dataType:
        return 'Categorical'
    elif ('int'in str(dataType)) & (unique == 2):
        return 'Binary'
    else :
        return 'Numerical'
