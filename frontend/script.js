//Constants------------------------------------------------
const api_URL = 'https://digit-classifier-fw6f.onrender.com'
const debugging_URL = 'http://127.0.0.1:5000'
const waitress_debugging_URL = 'http://localhost:8080'
const serve_URL = api_URL

//Helper functions ----------------------------------------
function setStroke(ctx){
    ctx.strokeStyle = 'white';
    ctx.lineWidth = Math.floor(size/28) * 2.5
    ctx.lineCap = 'round';  
}
function getBoundingBox(canvas, ctx){
    /*
    Get bounding box of canvas image so that we can pad it to make the prediction more accurate
    */
    const width = canvas.width;
    const height = canvas.height;

    const imageData = ctx.getImageData(0, 0, width, height);
    const data = imageData.data;
    //data contains 1D array with 4 channels RGBA

    //find the top of the bounding box
    let top = 0
    for (i = 0; i < data.length; i+= 4){
        if (i % (width*4) == 0) top+=1
        //if pixel is white
        if (data[i]) break
    }

    //find left of bounding box
    let curr_left = 0
    let left = width

    for (i = 0; i < data.length; i+= 4){
        curr_left +=1
        //if we move to the next row of pixels
        if(i % (width*4) == 0){
            curr_left = 0
        }
        
        if(data[i]) {
            if(curr_left < left) { 
                left = curr_left
            }
        }
    }

    //find bottom of bounding box
    let curr_bottom = 0; 
    let bottom = 0;
    for (i = 0; i < data.length; i+= 4){
        if (i % (width*4) == 0) curr_bottom+=1
        //if pixel is white
        if (data[i]) bottom = curr_bottom
    }

    //find the right of bounding box
    let curr_right = 0
    let right = 0

    for (i = 0; i < data.length; i+= 4){
        curr_right +=1
        //if we move to the next row of pixels
        if(i % (width*4) == 0){
            curr_right= 0
        }
        
        if(data[i]) {
            if(curr_right > right) { 
                right = curr_right
            }
        }
    }
    
    //Draw bounding box

    ctx.lineWidth = 1
    ctx.strokeStyle = 'green'

    ctx.rect(left-1, top-1, right-left + 2, bottom-top +2)
    ctx.stroke()
    setStroke(ctx)

    return {left, right, top, bottom}
}


function getBoundingBoxDataURL(canvas, ctx) {

    const boundingBox = getBoundingBox(canvas, ctx);
    const width = boundingBox.right - boundingBox.left;
    const height = boundingBox.bottom - boundingBox.top;

    //size x size canvas
    const padding = Math.floor(Math.max(width, height)/20) *4

    // Create a temporary canvas for cropping
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');

    //adjust padding to make sure image is square
    let width_padding
    let height_padding
    if(height > width){
        width_padding = padding + height-width
        height_padding = padding
    }
    else{
        height_padding = padding + width-height
        width_padding = padding
    } 

    // Set the size of the temporary canvas to the size of the bounding box + the padding
    tempCanvas.width = width + width_padding;
    tempCanvas.height = height + height_padding;

    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0,0,tempCanvas.width,tempCanvas.height)
    setStroke(tempCtx)

    // Draw the bounding box area onto the temporary canvas
    tempCtx.drawImage(
      canvas,
      boundingBox.left, boundingBox.top, width, height,
      width_padding/2, height_padding/2, width, height
    );

    // Get the Data URL for the cropped region
    const dataURL = tempCanvas.toDataURL('image/png');

    document.getElementById('test').href = dataURL
    return dataURL;
  }


function isMobileDevice() {
    return /Mobi|Android|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

//Main Script---------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {    

    const canvas = document.getElementById('drawingCanvas');
    if (isMobileDevice()){
        size = screen.width*0.75;  
    }
    else{
        size = screen.width*0.25; 
    }
    
    canvas.width = size; 
    canvas.height = size; 
    const ctx = canvas.getContext('2d');

    let drawing = false;


    //drawing context variables
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,canvas.width,canvas.height)

    setStroke(ctx)

    //mouse listeners on canvas
    canvas.addEventListener('mousedown', () => {drawing = true;});
    canvas.addEventListener('mouseup', ()=> {drawing = false; ctx.beginPath();});
    canvas.addEventListener('mousemove', draw);

    canvas.addEventListener('pointerdown', ()=> {drawing = true; ctx.beginPath();}); 
    canvas.addEventListener('pointerup', ()=>{drawing = false; ctx.closePath();} );
    canvas.addEventListener('pointerleave', ()=>{drawing = false})
    canvas.addEventListener('pointermove', draw)

    function draw(event){
        if(!drawing) return; 

        //Get the domRect (rectangle) objectof the canvas
        const rect = canvas.getBoundingClientRect();
        //X and Y positions on the canvas
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;

        //draw on the canvas

        //create line from previous x, y to current
        ctx.lineTo(x, y);
        //draw line with current stroke style
        ctx.stroke();

        //begin new path at current x,y
        ctx.beginPath();
        ctx.moveTo(x,y)
    }

    //clear button clears canvas
    document.getElementById('clearButton').addEventListener('click', () =>{
        ctx.fillRect(0,0, canvas.width, canvas.height)
        document.getElementById('prediction').innerHTML = ''
        document.getElementById('pred-div').hidden = true
        setStroke(ctx)
        document.getElementById('classifyButton').hidden = false
    })

    //classify button saves pixel data then sends it to the server with an http request
    document.getElementById('classifyButton').addEventListener('click', ()=>{

        const spinner = document.getElementById('spinner')
        const buttons = document.getElementById('buttons')
        spinner.hidden = false
        buttons.hidden = true
        
       //Get bounding box of number then add padding, then get pixel data
       imgData = getBoundingBoxDataURL(canvas, ctx)

       fetch(serve_URL +'/classify',{
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            image: imgData
        })})
        .then(response => {
            if(!response.ok){
                throw new Error('Network response was not ok')
            }
            return response.json();
        })
        .then(data =>{
            spinner.hidden = true
            buttons.hidden = false
            document.getElementById('classifyButton').hidden = true

            document.getElementById('pred-div').hidden = false

            prediction  = data['prediction']
            pred_element = document.getElementById('prediction')
            pred_element.innerHTML = 'You drew the number ' + prediction
        })
        .catch(error =>{
            console.error('there was a problem with the fetch operation:', error)
        })
    })
}); 