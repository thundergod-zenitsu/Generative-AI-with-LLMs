(function(){
    const container = document.getElementById('timeline');
    const data = new vis.DataSet(
        sales
        .filter((sale) => sale.product === 'carrots')
        .map((sale, i) => ({
            id : i,
            start : sale.date,
            content : `
            <table>
                <tr>
                    <td>Product:</td>
                    <td>${sale.product.substr(0,1).toUpperCase()+sale.product.substr(1)}</td>
                </tr>
                <tr>
                    <td>Sale:</td>
                    <td>${sale.sales}</td>
                </tr>
            </table>`
        }))
    );

    new vis.Timeline(container,data)
})();