firstFive = d3.json("/first_five")
latestData = d3.json("/get_latest_data")

// Function initializes the dashboard.
function init() {

    console.log('testing')

    latestData.then((data) => {
        console.log(data)
    });
    
    firstFive.then((data) => {
        console.log(data)
    });
}

// Initialize the dashboard
init();

