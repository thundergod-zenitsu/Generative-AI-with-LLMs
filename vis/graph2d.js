(function () {
    const container = document.getElementById('graph2d');
    const groups = new vis.DataSet();
    groups.add({
        id : 'carrots',
        content : 'carrots',
        options : {
            drawPoints : false,
            style: 'bar'
        }
    })
    const items = sales
        .filter((sale) => sale.product === 'carrots')
        .map((sale) => ({
            x: sale.date,
            y: sale.sales,
            group: 'carrots',
        }));
    const options = {
        defaultGroup : '',
        start : '2020-06-01',
        end : '2020-06-07',
        legend : true
    };
    new vis.Graph2d(container, items, groups, options);
})();