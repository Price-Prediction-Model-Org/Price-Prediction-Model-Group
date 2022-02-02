d3.json("/bitcoin_daily_data").then(data=>{
    console.log(data);

       
    var formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
    });
    
    
    ///////////////////////////----------------TABLE------------///////////////////////////////

    var openprice=formatter.format(data[(data.length)-1].open);
    var closeprice=formatter.format(data[(data.length)-1].close);
    var allhigh=formatter.format(Math.max(...data.map(row => row.high)));///... spreads array
    var alllow=formatter.format(Math.min(...data.map(row=>row.low)));
    var volume=formatter.format(data[(data.length)-1].volumefrom);

    d3.select("#openprice").text(openprice);
    d3.select("#closeprice").text(closeprice);
    d3.select("#high").text(allhigh);
    d3.select("#low").text(alllow);
    d3.select("#volume").text(volume);

    ////////////////////////////////------------CANDLESTICK PLOT----------//////////////////////////////////
    
        
    var trace1 = {
       
        x:data.map(row => row.timestamp_date),
        close:data.map(row => row.close),
        high:data.map(row => row.high),
        low:data.map(row => row.low),
        open:data.map(row => row.open),
       
    // cutomise colors
         increasing: {line: {color: 'green'}},
         decreasing: {line: {color: 'red'}},

        type: 'candlestick',
        xaxis: 'x',
        yaxis: 'y'
    };
    var layout1 = {
        dragmode: 'zoom',
        showlegend: false,
        plot_bgcolor:"black",
        paper_bgcolor:"black",
        xaxis: {
            autorange: true,
            title: 'Date',
            rangeselector: {
                x: 0,
                y: 1.2,
                xanchor: 'left',
                font: {size:10},
                bgcolor: '#054323',
                buttons: [{
                    step: 'month',
                    stepmode: 'backward',
                    count: 1,
                    label: '1 month',
                    
                }, {
                    step: 'month',
                    stepmode: 'backward',
                    count: 6,
                    label: '6 months'
                }, {
                    step: 'all',
                    label: 'All dates'
                }]
            }
        },
        yaxis: {
            autorange: true
        }
         
    };
     
    var tracedata1 = [trace1];
       

    Plotly.newPlot('plotcandlestick', tracedata1, layout1);

////////////////////////----------------Price vs Date AREA Chart------------////////////////
   
    var trace2 = {
            x:data.map(row => row.timestamp_date),
            y: data.map(row => row.close),
            // setopacity: 0,
            // fillOpacity: 0.75,
             fill:'tozeroy',
        //    fill: 'tonexty',
            type: 'scatter',
            mode: 'lines',
           fillcolor: '#4df761'
            // fillcolor: '#0eda1f',///green   
           };               
                
          var tracedata2 = [trace2];          
  
    Plotly.newPlot('plotPrice', tracedata2,layout1);
});
 



//----------------------------------plotly Model prediction---------------------------
// d3.json("Crypto_past_year_Predictions.json").then(data => {
d3.json("Model.json").then(data => {
    // console.log(data);
///////////---trace for Predictions for past date---//////
   let trace3 = {
     x:data.map(row => row.dates),
     y: data.map(row => row.predictions),
        name: "Predictions",
        mode: 'lines',
        line: {
        color: 'red',
        width: 3
         }
    }
///////////---trace for real data for past date---//////
    let trace4 = {
      x:data.map(row => row.dates),
      y: data.map(row => row.real_prices),
      name: "Realdata",
      mode: 'lines',
      line: {
      color: 'blue',
      width: 3
      }
  }

///////////---trace for Predictions for future date---//////
    let trace5 = {
      x:data.map(row => row.Datetime),
      y: data.map(row => row.predictions),
      name: "Future ",
      mode: 'lines',
      line: {
      color: 'orange',
      width: 3
     }
    }


    let traceData3 = [trace3,trace4];

    var layout1 = {
      dragmode: 'zoom',
      showlegend: true,
      plot_bgcolor:"black",
      paper_bgcolor:"black",
      xaxis: {
          autorange: true,
          title: 'Date',
          rangeselector: {
              x: 0,
              y: 1.2,
              xanchor: 'left',
              font: {size:10},
              bgcolor: '#054323',
              buttons: [{
                  step: 'month',
                  stepmode: 'backward',
                  count: 1,
                  label: '1 month',
                  
              }, {
                  step: 'month',
                  stepmode: 'backward',
                  count: 6,
                  label: '6 months'
              }, {
                  step: 'all',
                  label: 'All dates'
              }]
          }
      },
      yaxis: {
          autorange: true
      }
       
  };
   
  //         // Render the plot to the div tag with id "plotplotmodelprediction"
     Plotly.newPlot("plotmodelprediction", traceData3, layout1);


 });