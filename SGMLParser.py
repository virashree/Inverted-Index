from bs4 import BeautifulSoup


class SGMLParser:

    def __init__(self,file):
        self.soup = BeautifulSoup(open(file).read(), 'html.parser')

    def readTitle(self):
        return self.soup.title.text.strip()

    def readBody(self):
        return self.soup.find('text').text.strip()

    def getDocNo(self):
        return self.soup.docno.text.strip()

