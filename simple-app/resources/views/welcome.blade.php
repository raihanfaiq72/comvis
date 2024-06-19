<!DOCTYPE html>
<html>
<head>
    <title>Run Python Script</title>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
</head>
<body>
    <button id="run-script">Run Python Script</button>
    <div id="output"></div>

    <script>
        $('#run-script').click(function() {
            $.ajax({
                url: '/python-run',
                type: 'GET',
                success: function(data) {
                    $('#output').html(data.output);
                },
                error: function(error) {
                    console.error("Error running script:", error);
                }
            });
        });
    </script>
</body>
</html>
