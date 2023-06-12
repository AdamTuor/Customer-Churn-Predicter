// Add console.log to check to see if our code is working.
console.log("app.js");

x = []

d3.csv("/static/data/Telco-Customer-Churn.csv").then(function (data) {
    console.log(data)

    femail = 0
    male = 0
    dependents = 0
    no_dependents = 0
    fiber_optic = 0
    dsl = 0
    no_internet = 0


    for (i = 0; i < data.length; i++) {
        if (data[i]["gender"] == "Female") {
            femail += 1
        }    
        else {
            male += 1
        }

        if (data[i]["Dependents"] == "Yes") {
            dependents += 1
        }    
        else {
            no_dependents += 1
        }

        if (data[i]["InternetService"] == "DSL") {
            dsl += 1
        }    
        else if (data[i]["InternetService"] == "Fiber optic") {
            fiber_optic += 1
        }
        else {
            no_internet += 1
        }
 
    }

    var data = [{
        values: [femail, male],
        labels: ['Female', 'Male'],
        type: 'pie'
      }];
      
      var layout = {
        height: 400,
        width: 500
      };
      
      Plotly.newPlot('pie01', data, layout);

      var data = [{
        values: [dependents, no_dependents],
        labels: ['Dependents', 'No Dependents'],
        type: 'pie'
      }];
      
      var layout = {
        height: 400,
        width: 500
      };
      
      Plotly.newPlot('pie02', data, layout);

      var data = [{
        values: [fiber_optic, dsl, no_internet],
        labels: ["Fiber optic", "DSL", "No Internet"],
        type: 'pie'
      }];
      
      var layout = {
        height: 400,
        width: 500
      };
      
      Plotly.newPlot('pie03', data, layout);


})