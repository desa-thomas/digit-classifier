const URL = "http://127.0.0.1:5000"

document.addEventListener('DOMContentLoaded', () => {    

    const canvas = document.getElementById('drawingCanvas');
    size = screen.width*0.3; 
    canvas.width = size; 
    canvas.height = size; 
    const ctx = canvas.getContext('2d');

    let drawing = false;

    //drawing context variables
    ctx.fillStyle = 'black';
    ctx.fillRect(0,0,canvas.width,canvas.height)

    ctx.strokeStyle = 'white';
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';  

    //mouse listeners on canvas
    canvas.addEventListener('mousedown', () => {drawing = true});
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
        console.log('here')
        ctx.fillRect(0,0, canvas.width, canvas.height)
    })

    //classify button saves pixel data then sends it to the server with an http request
    document.getElementById('classifyButton').addEventListener('click', ()=>{
        console.log('classify')
        //Encode image to base64
       const imgData =  canvas.toDataURL("image/png")

       fetch(URL +'/classify',{
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
            console.log('Data Received;', data);
            console.log()
        })
        .catch(error =>{
            console.error('there was a problem with the fetch operation:', error)
        })
    })
}); 