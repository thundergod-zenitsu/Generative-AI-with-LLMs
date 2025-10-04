(function(){
    const container = document.getElementById("graph3d");
    const data = new vis.DataSet();
    const productArray = [
        'carrots',
        'potatoes',
        'corn',
        'green beans',
        'apples',
        'peaches',
        'pears',
        'plums',
        'beef',
        'chicken',

    ];
    sales.forEach((sale) => {
        data.add({
            x : productArray.indexOf(sale.product),
            y : moment(sale.date).date(),
            z : sale.sales
        });
    });
    const options = {
        style : 'bar',
        width : '600px',
        height : '600px',
        showPerspective : true,
        showGrid : true,
        showShadow : true,
        keepAspectRatio : true,
        verticalRatio : 0.5,
        tooltip : function(point) {
            return `Sales: ${point.z}`;
        },
        xStep : 1,
        xValueLabel : function(value){
            return productArray[value];
        },
        yValueLabel : function(value){
            return moment('2020-06-01').add(value).format('MMM DD');
        }
    };
    new vis.Graph3d(container,data,options);
})();