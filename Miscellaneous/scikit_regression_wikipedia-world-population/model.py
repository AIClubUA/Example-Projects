
#need python 3.6.x 64 bit, windowsX86-64 executable installer
#python -mpip install -U matplotlib
#pip install scipy
#pip install -U scikit-learn


import matplotlib.pyplot as plt
import pickle
import numpy as np 


#using np.polyval()
def interpolate(clean_dict):
    indv_res = []
    preds = 0
    deep = 0
    n = 3 # change this as you wish!
    for country in clean_dict:
        if deep < 10: #only do the first 10
            X = np.array(list(range(0, len(clean_dict[country][:-1]))))
            y = np.array(clean_dict[country][:-1])
            #print(X)
            #print(y)
            actual = clean_dict[country][-1]
            p1 = np.polyfit(X, y, n) #here the order from line 18 is applied
            pred = np.polyval(p1, len(clean_dict[country]))
            acc = pctAccurate(pred, actual) #do definition as come upon it
            preds += acc
            indv_res.append([country, acc])

            xp = np.linspace(0, 15, 100)
            plt.plot(X, y, 'o', xp, np.polyval(p1, xp), 'r:')
            plt.show()
        
        deep+=1
    
    print("\n\nPolynomial Interpoation Accuracy:", preds/len(indv_res), "\n\n")


def bayesRidge(clean_dict):
    from sklearn import linear_model
    indv_res = []
    preds = 0

    for country in clean_dict.keys():
        try:
            #print these out to see exactly what the data is
            X = [clean_dict[country][:-2]]          # training X values
            y = [clean_dict[country][-2]]           # training targets
            X_plot = [clean_dict[country][1:-1]]    # prediction X values
            actual = [clean_dict[country][-1]]      # actual value to gage accuracy
            
            """
            print('--------------------------')
            print(X)
            print(y)
            print('-----')
            """
            
            clf = linear_model.BayesianRidge()
            clf.fit(X, y)
            y_plot = clf.predict(X_plot)

            #print(y_plot)
            each_pred = pctAccurate(actual[0], y_plot[0])
            """
            if each_pred < .8:
                print(country, each_pred, actual[0], y_plot[0]) #bae
            """
            
            indv_res.append([country, each_pred])
            preds += each_pred
        except:
            print()

    print("\n\nBayesian Ridge Accuracy:", preds/len(indv_res))

def pctAccurate(actual, prediction):
    return 1 - abs(actual - prediction)/prediction


def main():
    raw_frame = pickle.load(open( "wiki_list_frame.dat", "rb" )) # reads in the pickle list list

    #---------------------- Making Data Frames -----------------------------------------------------
    clean_dict = {}
    clean_frame = []
    for i in raw_frame:
        country = i[0]
        buffer = [country] #initialized buffer with string country name
        intbuff = [] # holder for the data that has been converted to int from string
        for j in i[1:]:
            intbuff.append(int(j.replace(',', ''))) #need to replace ',' for typecasting to int to work
            #buffer.append(int(j.replace(',', ''))) #here we see the disparity of data, so need to normalize
        clean_dict[i[0]] = []
        for j in intbuff:
            normed = j/sum(intbuff) # normalizing the population ratios to itself, because some countries have very large and small pops so need growth ratios
            buffer.append(normed)
            clean_dict[i[0]].append(normed)
        
        clean_frame.append(buffer)

    #------------------- Optional Plotting --------------------------------------------------------------
    #------------ Good to visualize the data to see what shapes we are working with
    #plt.plot(clean_dict['Yemen'], 'ro')
    count = 0
    for country in clean_dict.keys():
        #plt.axis([0, 15, 0, 1])
        if count < 10:
            print(country)
            plt.plot(clean_dict[country])
            print('--------------')
            plt.show()
        count+=1

    #------------------ Here we see that the majority of popultions are either linear or quadratic growth, 
    #------------------ so first going to try some polynomial curve fitting

    interpolate(clean_dict)

    #------------------ Fairly decent predictions, but if plotting different order values, see that accuracy drops dramatically
    #------------------ Next will want to try bayesian ridge regression form sci-kit learn

    bayesRidge(clean_dict)

    #------------------ Slightly better results w/ baes ridge


if __name__ == '__main__':
    main()

