firstFive = d3.json("/first_five")

// Function initializes the dashboard.
function init() {

    console.log('testing')
    
    firstFive.then((data) => {
        console.log(data)
    });
}

// Initialize the dashboard
init();

