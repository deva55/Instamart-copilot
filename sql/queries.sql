--Q1
with cte as(
select customer_id, sum(order_amount) as total_order from orders where order_amount > 500 and status='completed' group by customer_id having sum(order_amount)>8000)
select * from cte; -- replace with count(*) to check 30

--Q2
with cte2 as(
select a.city, b.category, sum(c.order_amount) as revenue from 
    customers a join orders c on a.customer_id = c.customer_id 
        join products b on b.product_id = c.product_id 
where c.status= 'completed' 
group by a.city, b.category)
select * from cte2; --select sum(revenue) from cte2 o/p verified witht one in pandas eda.py

--Q3
select * from products where price_inr > (select avg(price_inr) from products); --replace with count(*) to get the count verified i.e.11 here

--Q4
with cte4 as(
    select customer_id from orders where status  in ('completed')
)
select a.customer_id from customers a left join cte4 b on a.customer_id = b.customer_id WHERE b.customer_id IS NULL; --count(*) checked 25 customers

--Q5
with cte5_1 as(
    select a.customer_id, sum(b.order_amount) as total_spend from customers a join orders b on a.customer_id = b.customer_id where b.status = 'completed' group by a.customer_id
),
cte5_2 as(
    select avg(total_spend) as avg_tot_spend from cte5_1
)
select customer_id, round(total_spend,2) as ts_above_avg from cte5_1 a where total_spend > (select avg_tot_spend from cte5_2); --count(customer_id) verified 550

--Q6
with cte6 as(
    select a.customer_id, sum(b.order_amount) as total_spend from customers a join orders b on a.customer_id = b.customer_id where b.status = 'completed' group by a.customer_id
)

select a.customer_id,
        a.total_spend,
        b.city,
        rank() over (partition by b.city order by a.total_spend desc) as rk, 
        dense_rank() over (partition by b.city order by a.total_spend desc) as dense_rk,
        row_number() over (partition by b.city order by a.total_spend desc) as rn
        from cte6 a join customers b
        on a.customer_id = b.customer_id
        limit 10;

--select city, total_spend from cte6_2 where rk=1 order by total_spend desc; 


--Q7
--select * from orders limit 5;
with cte7 as(
        select SUBSTR(order_ts, 1, 10) AS day, sum(order_amount) as revenue from orders where status = 'completed' group by SUBSTR(order_ts, 1, 10)
)

select day, revenue, sum(revenue) over (order by day) as cumulative_revenue from cte7 order by day limit 10;