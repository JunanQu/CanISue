from . import *
from app.irsystem.models.helpers import *
from app.irsystem.models.helpers import NumpyEncoder as NumpyEncoder

project_name = "Team Rob's Chili"
net_id = "jq77, zs92, ijp9, mlc294, ns739"


@irsystem.route('/', methods=['GET'])
def search():
    query = request.args.get('search')
    print(query)
    output_message = ''
    if not query:
        data = []
    else:
        data = range(2)
        output_message_1 = "Your search: " + query
        if(len(data) >= 3):
            output_message_2 = 'Here are the top 3 related cases'
        else:
            output_message_2 = 'Here are the top {n:.0f} related cases'.format(
                n=len(data))
        output_message = output_message_1+' \n '+output_message_2
    return render_template('search.html', name=project_name, netid=net_id, output_message=output_message, data=data)
