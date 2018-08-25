

import requests
import bs4 as bs

import pickle
import csv

# Functions are for saving data later, not important
def pickleWrite(data, filename):
    if '.dat' not in filename:
        filename = filename + '.dat'
    pickle.dump(data, open(filename, 'wb'), -1)
    print('[+] Successfully saved', filename, '[+]\n')


def csvWriteRow(yuuge_list, filename):
    if '.csv' not in filename:
        filename = filename + '.csv'

    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in yuuge_list:
            writer.writerow(line)
    
    print('[+] Successfully exported data to', filename, '[+]\n')


#----------------------------------------------------------------------------------------------
# Main sequence

# retrieve the page
url = "https://en.wikipedia.org/wiki/List_of_countries_by_past_population_(United_Nations,_estimates)"
r = requests.get(url) #basic get request to attain html
soup = bs.BeautifulSoup(r.text, 'html5lib') #or 'lxml', parses the html for procesing

# Organizer headers
raw_head = soup.find_all("th")[1:15] #indexes the parsed page, searching for "th" tag, returning 1:15 tags
headers = ['Country'] #makes list to recieve headers
for line in raw_head:
    header = str(line).strip('<th>').strip('</th>')
    headers.append(header)
print(headers)

list_frame = [headers] # Create the frame to hold full data

#populate the main frame
entries = soup.find_all("tr") #find "tr" tag containing data we want
for i in entries:
    try:
        #grabs title
        title = i.a['title'] # where a is the tag we are looking for and 'title' is embedded in the tagline
        print(title)
        buffer = [title] #initializes buffer of each line of data on the page

        #grabs population stuff
        pop_data = i.find_all("td")[1:] # index 1: to drop first element
        for data_line in pop_data:
            buffer.append(data_line.get_text()) #get_text() replaces the .strip() functions on line 38

        print(buffer)
        print('-------------------')
        #only adds if long enough
        if len(buffer) == 15: #length of good buffer is 15
            list_frame.append(buffer)
    except:
        print('------')

#print out the resultant frame
for i in list_frame:
    print(i)

#saves the data in two formats
pickleWrite(list_frame, 'wiki_list_frame') #for reading back in to python in same list-list structure
csvWriteRow(list_frame, 'wiki_list_frame') #for visualising in excel