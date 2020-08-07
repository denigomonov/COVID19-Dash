#libs
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from plotly.subplots import make_subplots
from heapq import nlargest
###############################################################################

#laoding datasets
virus_cases_df=pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_confirmed_usafacts.csv')
virus_deaths_df=pd.read_csv('https://usafactsstatic.blob.core.windows.net/public/data/covid-19/covid_deaths_usafacts.csv')

#function to self-transform data
def data_transform (x):

    #groupping observations by State
    grouped_data=x.groupby(['State']).sum()
    #modifying & cleaning
    grouped_data.drop(grouped_data.columns[0:2], axis=1, inplace=True)
    grouped_data.reset_index(inplace=True)
    #transpose
    tran_data=grouped_data.transpose()
    #modifying & cleaning
    tran_data.reset_index(inplace=True)
    tran_data.columns=tran_data.iloc[0]
    tran_data.rename({'State':'Date'}, axis=1, inplace=True)
    tran_data.drop(tran_data.index[0], inplace=True)

    return tran_data
#call
virus_cases_tran_df=data_transform(virus_cases_df)
virus_deaths_tran_df=data_transform(virus_deaths_df)

#function to self-select top ten states
def top_ten_states(x):

    #empty dic, list
    max_case_d={}
    state_name=['Date']
    #getting top sum observations
    for i in x.columns[1:]:
        max_val=x[i].max()
        max_case_d[i]=max_val
    #getting top 10 from dic
    ten_highest=nlargest(10, max_case_d, key=max_case_d.get)
    #populating names of states to list
    for i in ten_highest:
        state_name.append(i)
    #returning new df
    virus_df_new=x[state_name]

    return virus_df_new
#call
virus_cases_df_top_ten=top_ten_states(virus_cases_tran_df)
virus_deaths_df_top_ten=top_ten_states(virus_deaths_tran_df)

def total_cases_allstates(x):

    #grouping data
    grouped_data=x.groupby(['State']).sum()
    grouped_data.reset_index(inplace=True)
    #only using needed columns
    state_lastcases_df=grouped_data.iloc[:, [0, -1]]
    state_lastcases_df.rename(columns={state_lastcases_df.columns[1]: 'Cases'}, inplace=True)
    asc_df=state_lastcases_df.sort_values(by='Cases', ascending=False)
    return asc_df
#call
virus_total_cases_all_df=total_cases_allstates(virus_cases_df)
virus_total_deaths_all_df=total_cases_allstates(virus_deaths_df)

def total_cases(x):

    #grouping data
    grouped_data=x.groupby(['State']).sum()
    grouped_data.reset_index(inplace=True)
    #only using needed columns
    state_lastcases_df=grouped_data.iloc[:, [0, -1]]
    state_lastcases_df.rename(columns={state_lastcases_df.columns[1]: 'Cases'}, inplace=True)
    state_lastcases_ten_df=state_lastcases_df.nlargest(10, 'Cases')
    asc_df=state_lastcases_ten_df.sort_values(by='Cases', ascending=True)
    return asc_df

#call
virus_total_cases_df=total_cases(virus_cases_df)
virus_total_deaths_df=total_cases(virus_deaths_df)

def daily_total(x):

    #grabbing columns into a list
    col_list=x.columns[1:].tolist()
    #summing observations into a new column
    x['Daily_Total']=x[col_list].sum(axis=1)
    new_df=x.iloc[:, [0, -1]]

    return new_df
#call
daily_cases_df=daily_total(virus_cases_tran_df)
daily_deaths_df=daily_total(virus_deaths_tran_df)

daily_cases_avg=daily_cases_df.Daily_Total.tolist()
daily_cases_sma_avg_df=pd.DataFrame(daily_cases_avg)
daily_cases_sma_avg_df['SMA_Cases']=round(daily_cases_sma_avg_df.rolling(window=3).mean(), 0)
daily_deaths_avg=daily_deaths_df.Daily_Total.tolist()
daily_deaths_sma_avg_df=pd.DataFrame(daily_deaths_avg)
daily_deaths_sma_avg_df['SMA_Cases']=round(daily_deaths_sma_avg_df.rolling(window=3).mean(), 0)

###############################################################################

#viz funcs
def first_graph():
    #figure
    cases_deaths_fig=make_subplots(rows=2, cols=1, vertical_spacing=0.08)

    #setting colors
    state_case_colors=['rgb(0, 255, 204)', 'rgb(0, 230, 184)', 'rgb(0, 204, 163)',
                  'rgb(0, 179, 143)', 'rgb(0, 153, 122)', 'rgb(0, 128, 102)',
              'rgb(0, 102, 82)', 'rgb(0, 77, 61)', 'rgb(0, 51, 41)',
              'rgb(0, 26, 20)']
    state_death_colors=['rgb(255, 0, 0)', 'rgb(230, 0, 0)', 'rgb(204, 0, 0)',
                    'rgb(179, 0, 0)', 'rgb(153, 0, 0)', 'rgb(128, 0, 0)',
                    'rgb(102, 0, 0)', 'rgb(77, 0, 0)', 'rgb(51, 0, 0)',
                    'rgb(26, 0, 0)']

    #assigning colors for cases
    state_cases_name=[]
    for i in virus_cases_df_top_ten.columns[1:]:
        state_cases_name.append(i)
    state_cases_color_dic=dict(zip(state_cases_name, state_case_colors))
    #assigning colors for deaths
    state_deaths_name=[]
    for i in virus_deaths_df_top_ten.columns[1:]:
        state_deaths_name.append(i)
    state_deaths_color_dic=dict(zip(state_deaths_name, state_death_colors))

    #adding case traces to the plot 1
    for i in virus_cases_df_top_ten.columns[1:]:
        cases_deaths_fig.add_trace(go.Scatter(x=virus_cases_df_top_ten.Date, y=virus_cases_df_top_ten[i], name=i, mode='lines',
                                 line=dict(color=state_cases_color_dic.get(i), width=3), fill='tozeroy',
                                 hovertemplate='<b>Date:</b> %{x} <br><b>Cases:</b> %{y}'),
                                 row=1, col=1
                             )
    #adding death tracess to the plot 2
    for i in virus_deaths_df_top_ten.columns[1:]:
        cases_deaths_fig.add_trace(go.Scatter(x=virus_deaths_df_top_ten.Date, y=virus_deaths_df_top_ten[i], name=i, mode='lines',
                                 line=dict(color=state_deaths_color_dic.get(i), width=3), fill='tozeroy',
                                 hovertemplate='<b>Date:</b> %{x} <br><b>Deaths:</b> %{y}'),
                                 row=2, col=1
                                 )

    #customizing figure
    cases_deaths_fig.update_layout(
        title_text=' COVID-19 Cases & Deaths Area Growth in TOP-10 States',
        title_font_family='Arial',
        title_font_color='rgb(74, 160, 207)',
        title={
            'x':0.5,
            'y':0.98
        },
        margin=dict(
            l=25,
            r=25,
            t=40,
            b=25
        ),
        legend=dict(
            bgcolor='rgb(0, 51, 77)',
            bordercolor='rgb(0, 68, 102)',
            borderwidth=2,
            font=dict(
                family="Arial",
                size=10,
                color='rgb(74, 160, 207)'
            )
        ),
        showlegend=True,
        plot_bgcolor='rgb(0, 51, 77)',
        paper_bgcolor='rgb(0, 17, 26)'
    )
    cases_deaths_fig.update_xaxes(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(74, 160, 207)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
                family='Arial',
                size=10,
                color='rgb(74, 160, 207)'
        ),
        dtick=16,
        row=1,
        col=1
    )
    cases_deaths_fig.update_xaxes(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(74, 160, 207)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
                family='Arial',
                size=10,
                color='rgb(74, 160, 207)'
        ),
        dtick=16,
        row=2,
        col=1
    )
    cases_deaths_fig.update_yaxes(
        title_text='COVID-19 Cases Count',
        title_standoff=1,
        title_font_color='rgb(74, 160, 207)',
        title_font_size=12,
        showline=True,
        showgrid=True,
        gridwidth=0.1,
        gridcolor='rgb(0, 68, 102)',
        linecolor='rgb(74, 160, 207)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(74, 160, 207)'
        ),
        zeroline=False,
        row=1,
        col=1
    )
    cases_deaths_fig.update_yaxes(
        title_text='COVID-19 Deaths Count',
        title_standoff=8,
        title_font_color='rgb(74, 160, 207)',
        title_font_size=12,
        showline=True,
        showgrid=True,
        gridwidth=0.1,
        gridcolor='rgb(0, 68, 102)',
        linecolor='rgb(74, 160, 207)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(74, 160, 207)'
        ),
        zeroline=False,
        row=2,
        col=1
    )

    return cases_deaths_fig

#second graph func
def third_graph():

    fig_sma=go.Figure()

    fig_sma.add_trace(go.Indicator(
        mode = "number+delta",
        value = daily_cases_sma_avg_df[0].iloc[-1],
        title = {"text": "<span style='font-size:20;color:rgb(0, 255, 204)'>Total Cases</span>"},
        number = dict(
            prefix="# ",
            font=dict(
                size=50
            )
        ),
        delta = dict(
            position="top",
            reference=daily_cases_sma_avg_df.SMA_Cases.iloc[-1],
            relative=False,
            font=dict(
                size=30
            )
        ),
        domain = {'x': [0, 1], 'y': [0.5, 1]})
    )

    fig_sma.add_trace(go.Indicator(

        mode = "number+delta",
        value = daily_deaths_sma_avg_df[0].iloc[-1],
        title = {"text": "<span style='font-size:20;color:rgb(255, 0, 0)'>Total Deaths</span>"},
        number = dict(
            prefix="† ",
            font=dict(
                size=50
            )
        ),
        delta = dict(
            position="top",
            reference=daily_deaths_sma_avg_df.SMA_Cases.iloc[-1],
            relative=False,
            font=dict(
                size=30
            )
        ),
        domain = {'x': [0, 1], 'y': [0, 0.5]})
    )

    fig_sma.update_layout(
        title_text='COVID-19 3-Num SMA',
        title_font_family='Arial',
        title_font_color='rgb(74, 160, 207)',
        #title_font_size=9,
        title={
            'x':0.5,
            'y':0.98,
        },
        showlegend=False,
        margin=dict(
            l=0,
            r=0,
            t=30,
            b=0
        ),
        paper_bgcolor='rgb(0, 17, 26)',
        font=dict(
                family="Arial",
                color='rgb(255, 204, 51)'
        )
    )

    return fig_sma

#third func graph
def second_graph():
    fig_bar=go.Figure()

    state_case_colors=['rgb(0, 255, 204)', 'rgb(0, 230, 184)', 'rgb(0, 204, 163)',
                  'rgb(0, 179, 143)', 'rgb(0, 153, 122)', 'rgb(0, 128, 102)',
              'rgb(0, 102, 82)', 'rgb(0, 77, 61)', 'rgb(0, 51, 41)',
              'rgb(0, 26, 20)']
    state_case_colors.reverse()
    state_death_colors=['rgb(255, 0, 0)', 'rgb(230, 0, 0)', 'rgb(204, 0, 0)',
                    'rgb(179, 0, 0)', 'rgb(153, 0, 0)', 'rgb(128, 0, 0)',
                    'rgb(102, 0, 0)', 'rgb(77, 0, 0)', 'rgb(51, 0, 0)',
                    'rgb(26, 0, 0)']
    state_death_colors.reverse()
    fig_bar.add_trace(go.Bar(x=virus_total_cases_df.Cases, y=virus_total_cases_df.State,
                         marker_color=state_case_colors, marker_line_width=0,
                         marker_line_color='rgb(0, 68, 102)', orientation='h',
                         hovertemplate='<b>State:</b> %{x} <br><b>Cases:</b> %{y}', name=''
                         )
                 )
    fig_bar.add_trace(go.Bar(x=virus_total_deaths_df.Cases, y=virus_total_deaths_df.State,
                         marker_color=state_death_colors, marker_line_width=0,
                         marker_line_color='rgb(0, 68, 102)', orientation='h',
                         hovertemplate='<b>State:</b> %{x} <br><b>Cases:</b> %{y}', name='',
                         visible=False
                         )
                 )

    fig_bar.update_layout(
        showlegend=False,
        margin=dict(
            l=5,
            r=10,
            t=40,
            b=25
        ),
        plot_bgcolor='rgb(0, 51, 77)',
        paper_bgcolor='rgb(0, 17, 26)',
        updatemenus = list([
        dict(
             buttons=list([
                dict(label = 'Cases',
                     method = 'update',
                     args = [{'visible': [True, False]},
                            ]),
                dict(label = 'Deaths',
                     method = 'update',
                     args = [{'visible': [False, True]},
                            ])
            ]),
            showactive=True,
            y=1.111,
            x=1,
            font=dict(
                family="Arial",
                size=14,
                color='rgb(74, 160, 207)'
            )
        )
        ])
    )

    fig_bar.update_xaxes(
        showline=True,
        showgrid=False,
        linecolor='rgb(74, 160, 207)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
                family='Arial',
                size=10,
                color='rgb(74, 160, 207)'
        )
    )

    fig_bar.update_yaxes(
        #showline=True,
        # showgrid=True,
        #gridwidth=0.1,
        # gridcolor='rgb(0, 68, 102)',
        # linecolor='rgb(74, 160, 207)',
        # linewidth=2,
        # ticks='outside',
        # tickfont=dict(
        #     family='Arial',
        #     size=12,
        #     color='rgb(74, 160, 207)'
        # ),
        visible=False,
        zeroline=False
    )

    return fig_bar
###############################################################################



#running app
app = dash.Dash(__name__)

app.layout = html.Div([
    #title div
    html.Div(className='div_title',
    children=[
        #title
        html.H1(className='title',
         children=['COVID-19 in U.S.']),
        #line
        html.Hr(className='title_line'),
        #title of first ul
        html.P(className='title_descrp',
         children=['Prevent Getting Sick:']),
        #bullet points
        html.Ul(className='title_list',
         children=[
            html.Li('There is currently no vaccine to prevent coronavirus disease 2019 (COVID-19).'),
            html.Li('The best way to prevent illness is to avoid being exposed to this virus.'),
            html.Li('The virus is thought to spread mainly from person-to-person.'),
            html.Li('Some recent studies have suggested that COVID-19 may be spread by people who are not showing symptoms.')
         ]),
         #title of second ul
         html.P(className='title_descrp',
          children=['Everyone Should Do To Stay Safe:']),
         #bullet points
         html.Ul(className='title_list2',
          children=[
             html.Li('Wash your hands often with soap and water for at least 20 seconds especially after you have been in a public place, or after blowing your nose, coughing, or sneezing.'),
             html.Li('Always cover your mouth and nose with a tissue when you cough or sneeze or use the inside of your elbow and do not spit.'),
             html.Li('Wear a mask in public settings and when around people who don’t live in your household, especially when other social distancing measures are difficult to maintain.'),
             html.Li('Clean AND disinfect frequently touched surfaces daily; this includes tables, doorknobs, light switches, countertops, handles, desks, phones, keyboards, toilets, faucets, and sinks.')
          ])
    ]),
    #line
    html.Hr(className='title_line'),
    #first graph div
    html.Div(className='first_graph_div',
    children=[
        dcc.Graph(
        className='first_graph',
        figure=first_graph())
    ]),
    html.Div(className='second_graph_div',
    children=[
        dcc.Graph(
        className='second_graph',
        figure=second_graph(),
        config={
        'displayModeBar': False
        }
        )
    ]),
    html.Div(className='third_graph_div',
    children=[
        dcc.Graph(
        className='third_graph',
        figure=third_graph()
        )
    ])
])

if __name__ == '__main__':
    app.run_server(debug=False, port=8051)
