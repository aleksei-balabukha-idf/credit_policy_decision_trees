create table service_cases (
    case_id     string,
    client_id   int,
    case_dttm   timestamp,
    escalated   boolean,
    resolved    boolean,
    case_type   string
);
insert into service_cases values
    ('CS01', 100, '2025-01-05 10:00', true,  false, 'billing'),
    ('CS02', 100, '2025-03-10 11:00', false, true,  'general'),
    ('CS03', 200, '2025-02-01 09:30', true,  true,  'billing'),
    ('CS04', 200, '2025-04-20 16:45', true,  false, null),
    ('CS05', 300, '2025-05-01 12:00', true,  true,  'tech'),
    ('CS06', 300, '2025-06-01 08:15', true,  false, 'spam'),
    ('CS07', 400, '2025-06-15 14:20', false,  false, 'tech'),
    ('CS08', 500, '2025-07-01 10:05', false, false, 'general')
    ('CS09', 500, '2025-07-01 10:15', false, true,  'general');
 
create table client_segments (
    client_id   int,
    segment     string,
    valid_from  date
);
insert into client_segments values
    (100, 'mass',    '2024-01-01'),
    (200, 'premium', '2024-01-01'),
    (300, 'mass',    '2024-01-01'),
    (400, 'mass',    '2024-01-01'),
    (400, 'premium', '2025-06-01'),
    (500, 'mass',    '2024-01-01');




на нужно собрать таблицу  - риск клиента (флаг риск True/False)
клиента считаем риском - если последний эскалированный кейс не решен






with ranked as (
    select
        client_id,
        escalated,
        resolved,
        row_number() over (
            partition by client_id
            order by case_dttm desc
        ) as rn
    from service_cases
    where
        coalesce(case_type, '') != 'spam'
        and escalated = 1
),
latest_escalation as (
    select client_id, resolved
    from ranked
    where
        rn = 1
)
select 
    client_id,
    segment,
    at_risk
from (
    select
        s.client_id,
        s.segment,
        case when l.resolved = false then 'yes' else 'no' end as at_risk,
        row_number() over (partition by s.client_id order by s.timestamp desc) as rn
    from client_segments s
    left join latest_escalation l on l.client_id = s.client_id)
where
    rn = 1
