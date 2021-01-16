from django.http import HttpResponse
from django.template import loader
from chatbot.utils.googlesearcher import scraper
from chatbot.utils.chatter import chatbot_response
import re


replacables = [
 'i', 'we', 'you', 'he', 'she', 'it',
 'they', 'me', 'us', 'her', 'him', 'will',
 'them', 'mine', 'ours', 'yours', 'was',
 'hers', 'his', 'theirs', 'my', 'is', 'not'
 'our', 'your', 'their', 'myself', 'are',
 'yourself', 'herself', 'himself', 'were',
 'itself', 'ourselves', 'yourselves', 'do',
 'themselves', 'all', 'another', 'any',
 'anybody', 'anyone', 'anything', 'both',
 'each', 'either', 'everybody', 'everyone',
 'everything', 'few', 'many', 'most', 'shall',
 'neither', 'nobody', 'none', 'no one', 'can'
 'nothing', 'one', 'other', 'others', 'should',
 'several', 'some', 'somebody', 'someone', 'where',
 'something', 'such', 'what', 'whatever', 'about',
 'which', 'whichever', 'who', 'whoever', 'could',
 'whom', 'whomever', 'whose', 'how', 'when',
]


# Create your views here.
def index(request):
    template = loader.get_template('chatbot/index.html')
    response = ''
    if request.method == 'POST':
        _input = request.POST.get('inputer')
        _input = str(_input)

        res, ints = chatbot_response(_input)

        print(res, ints)

        if len(ints)==1 and float(ints[0]['probability'])>0.8:
            response = res
        else:
            _input = _input.lower()
            _input = ''.join(re.findall('[a-zA-Z\n0-9 ]', _input))
            _input = _input.split(' ')
            input_ = ''
            for i in _input:
                if i in replacables:
                    pass
                else:
                    input_ = input_ + ' ' + i
            input_ = input_.replace('  ', ' ').strip()

            print(input_)

            response = scraper.searchQuery(input_)

    context = { 'key': response }
    print(response)
    return HttpResponse(template.render(context, request))

