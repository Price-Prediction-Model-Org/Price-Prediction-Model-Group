////////--------BTC DATA---//////////////////////
d3.json("/bitcoin_daily_data").then(btcdata=>{
    // console.log(btcdata);
    
////////--------ETH DATA---//////////////////////
d3.json("/ETH_daily_data").then(ethdata=>{
    // console.log(ethdata);   
    // console.log((btcdata[(btcdata.length)-1].volumefrom)+(btcdata[(btcdata.length)-1].volumeto));
    // console.log(btcdata);
    // console.log(ethdata);


    //////////////////////////------MARQUEE----------//////////////////////////////
    
    var formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
    });
    
    var Bitcloseprice= formatter.format(btcdata[(btcdata.length)-1].close);
    // var EthVar=formatter.format(12481);
     var ETHcloseprice=formatter.format(ethdata[(ethdata.length)-1].close);
    //  var Bitcloseprice=formatter.format(55555);

    d3.select("#whitetext1").text("Top Assets --> ");
    d3.select("#greentext1").text("BitCoin : ");
    d3.select("#redtext1").text(Bitcloseprice);
    d3.select("#whitetext2").text("  Ethereum :");
    d3.select("#redtext2").text(ETHcloseprice);

///////////////////////////----------------TABLE------------///////////////////////////////
    var formatter = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
    });
    var millionformatter=new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        notation: "compact",
        compactDisplay: "short"
    });
    /////DATA FOR BITCOIN////////////
    var assetname="Bitcoin";
    var closeprice=formatter.format(btcdata[(btcdata.length)-1].close);
    var volume=millionformatter.format((btcdata[(btcdata.length)-1].volumefrom)+(btcdata[(btcdata.length)-1].volumeto));
    var allhigh=formatter.format(Math.max(...btcdata.map(row => row.high)));///... spreads array
    var alllow=formatter.format(Math.min(...btcdata.map(row=>row.low)));

    // console.log(volume)

    d3.select("#assetname1").text(assetname);
    d3.select("#closeprice1").text(closeprice);
    d3.select("#volume1").text(volume);
    d3.select("#high1").text(allhigh);
    d3.select("#low1").text(alllow);
    /////DATA FOR ETHEREUM ///////////////////////

    var assetname="Ethereum";
    var closeprice=formatter.format(ethdata[(ethdata.length)-1].close);
    var volume=millionformatter.format((ethdata[(ethdata.length)-1].volumefrom)+(ethdata[(ethdata.length)-1].volumeto));
    var allhigh=formatter.format(Math.max(...ethdata.map(row => row.high)));///... spreads array
    var alllow=formatter.format(Math.min(...ethdata.map(row=>row.low)));


    d3.select("#assetname2").text(assetname);
    d3.select("#closeprice2").text(closeprice);
    d3.select("#volume2").text(volume);
    d3.select("#high2").text(allhigh);
    d3.select("#low2").text(alllow);


//////////////////////////-------COMPARISON PLOT-------///////////////////////////////
//d3.json("Model.json").then(data => {
    // console.log(data);
///////////---trace for Bitcoin Closing Price---//////
   let trace6 = {
     x:btcdata.map(row => row.timestamp_date),
     y: btcdata.map(row => row.close),
        name: "Bitcoin",
        mode: 'lines',
        line: {
        color: 'red',
        width: 3
         }
    }
///////////---trace for Ethereum Closing Price---//////
    let trace7 = {
      x:ethdata.map(row => row.timestamp_date),
      y: ethdata.map(row => row.close),
      name: "Ethereum",
      mode: 'lines',
      line: {
      color: 'blue',
      width: 3
      }
  }

///////////---trace for Tether Closing Price---//////
    // let trace8 = {
    //   x:data.map(row => row.Datetime),
    //   y: data.map(row => row.close),
    //   name: "Tether ",
    //   mode: 'lines',
    //   line: {
    //   color: 'orange',
    //   width: 3
    //  }
    // }


    let traceData4 = [trace6,trace7];

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
              buttons: [
                {
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
   
  //         // Render the plot to the div tag with id "plotcompare"
    Plotly.newPlot("plotcompare", traceData4, layout1);

});
});