from selenium import webdriver
import os



class BtsScraper:
    def __init__(self):
        options = webdriver.ChromeOptions()

        # options.binary_location = os.environ.get('GOOGLE_CHROME_BIN')
        # options.add_argument("--headless")
        # options.add_argument("--disable-dev-shm-usage")
        # options.add_argument("--no-sandbox")
        # self.driver = webdriver.Chrome(executable_path=os.environ.get('CHROMEDRIVER_PATH'), chrome_options=options)

        options.add_argument("--headless")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        self.driver = webdriver.Chrome(executable_path='static/chatbot_staticfiles/chromedriver', chrome_options=options)


    def searchQuery(self, query):

        browser = self.driver

        url = 'https://www.wikipedia.org/'

        browser.get(url)

        search_form = browser.find_element_by_id('searchInput')
        search_form.send_keys(query.strip().replace(' ', '_'))
        search_form.submit()

        print(browser.current_url)

        response = ''

        if 'https://en.wikipedia.org/wiki/Special:Search?search' in browser.current_url:
            response = "Ummm... I am not sure, but I think so: "

            browser.get(browser.current_url)

            try:
                content = browser.find_element_by_id('mw-content-text')\
                    .find_element_by_class_name('searchresults')\
                    .find_element_by_tag_name('ul')\
                    .find_element_by_tag_name('li')

                response = response + content.text
            except:
                response = "I don't know what you want me to say !"

        else:
            content = browser.find_element_by_id('mw-content-text')\
                .find_element_by_class_name('mw-parser-output')\
                .find_elements_by_tag_name('p')

            i=0
            while len(response) < 250 and i < len(content):
                response = str(response) + '\n' + str(content[i].text)
                i+=1

            if ('most commonly refers to' in response) and ('may also refer to' in response):
                content = browser.find_element_by_id('mw-content-text') \
                    .find_element_by_tag_name('ul') \
                    .find_element_by_tag_name('li')
                response = 'Here is what I found: ' + content.text

            if 'may refer to:' in str(response):
                content = browser.find_element_by_id('mw-content-text') \
                    .find_element_by_tag_name('div') \
                    .find_element_by_tag_name('ul')
                response = 'Here is what I found: ' + content.text


        return response


# ------------------------------------------------------------------------------------


scraper = BtsScraper()




