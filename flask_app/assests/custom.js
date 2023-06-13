document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('telecomForm').addEventListener('submit', function(event) {
      event.preventDefault();  // Prevent the default form submission behavior
  
      // Trigger the Dash callback manually
      var button = document.getElementById('predict-button');
      button.click();
    });
  });