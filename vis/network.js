(function () {
    const container = document.getElementById('network');

    let nextId = 0;
    const oneDaySales = sales.filter((s) => s.date === '2020-06-01');
    const categories = oneDaySales.reduce((categories, sale) => {
        if (!categories.find((c) => c.name === sale.category)) {
            categories.push({
                id: nextId++,
                name: sale.category,
            });
        }

        return categories;
    }, []);
    const subcategories = oneDaySales.reduce((subcategories, sale) => {
        if (!subcategories.find((s) => s.name === sale.subcategory)) {
            subcategories.push({
                id: nextId++,
                name: sale.subcategory,
                category: categories.find((cat) => cat.name === sale.category),
            });
        }
        return subcategories;
    }, []);

    const products = oneDaySales.reduce((products, sale) => {
        products.push({
            id: nextId++,
            name: sale.product,
            subcategory: subcategories.find(
                (subcat) => subcat.name === sale.subcategory
            ),
        });
        return products;
    }, []);

    const nodes = new vis.DataSet([
        ...products.map((p) => ({ id: p.id, label: p.name })),
        ...subcategories.map((s) => ({ id: s.id, label: s.name })),
        ...categories.map((c) => ({ id: c.id, label: c.name })),
    ]);

    const edges = new vis.DataSet([
        ...products.map((p) => ({ from: p.subcategory.id, to: p.id })),
        ...subcategories.map((s) => ({ from: s.category.id, to: s.id })),
    ]);

    const data = {
        nodes,
        edges,
    };

    new vis.Network(container, data, {});
})();