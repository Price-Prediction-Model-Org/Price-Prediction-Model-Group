firstFive = d3.json("/first_five")
get_predictions = d3.json("/model_predictions")

// Function initializes the dashboard.
function init() {

    console.log('testing')
    
    firstFive.then((data) => {
        console.log(data)
    });

    get_predictions.then((data) => {
        console.log(data)
    });

}

// Initialize the dashboard
init();

