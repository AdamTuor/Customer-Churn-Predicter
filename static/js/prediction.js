// Add console.log to check to see if our code is working.
console.log("prediction.js");

function make_pred(){
    console.log("make_pred function");

    let contract = document.getElementById("contract").value;
    let internet = d3.select("#internet").property("value");
    let paymentmethod = d3.select("#paymentmethod").property("value"); 
    let tenure = d3.select("#tenure").property("value");
     
    console.log("contract", contract);
    console.log("internetservice", internet);
    console.log("paymentmethod", paymentmethod);
    console.log("tenure", tenure);
        
    fetch("/predict",{
        method: "POST", 
        body: JSON.stringify({
            tenure: tenure,
            internetservice: internet,
            contract: contract,
            paymentmethod: paymentmethod
        }),
        headers:{
            "Content-type":"application/json;charset=UTF-8"

        }

    }).then(resp=>{
        return resp.json()
    }).then(resp=>{
        console.log(resp)
        // document.getElementById("prediction").innerHTML=resp.Prediction
        console.log("prediction= " + resp.Prediction);
        if (resp.Prediction=="Yes"){
            document.getElementById("prediction").innerHTML="Low Risk to Churn"
            document.getElementById("dummy").src= "/static/images/happy_face.jpg" 
        }
        else if (resp.Prediction=="No"){
            document.getElementById("prediction").innerHTML="High Risk to Churn"
            document.getElementById("dummy").src="/static/images/cry_face.png"
        }
    })
}